"""
panel_strategy.py — Торговая панель, подключённая напрямую к Nautilus Trader.

Работает ПАРАЛЛЕЛЬНО с любыми другими стратегиями (SMCStrategy, etc.)
Добавить в flask_server.py:
    from panel_strategy import PanelStrategy
    panel = PanelStrategy(instrument_id, gui_queue, manual_order_queue)
    node.trader.add_strategy(panel)

Возможности (через NT API):
  - Все типы ордеров: Market / Limit / StopMarket / StopLimit / MIT / LIT
  - Time In Force: GTC / IOC / FOK
  - Флаги: post_only, reduce_only
  - SL/TP как OCO после исполнения входа
  - Реальный баланс и PnL из portfolio/account (Binance)
  - Расчёт комиссии из instrument.maker_fee / taker_fee
  - Оценка слипажа через instrument.price_increment (tick_size)
  - Paper-режим: симуляция позиции, SL/TP по ценовым тикам
  - Cancel All — через self.cache.orders_open
"""
import queue as _queue
from decimal import Decimal

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.enums import (
    OrderSide, TimeInForce, TriggerType,
)
from nautilus_trader.model.events import (
    OrderFilled, OrderCanceled, OrderRejected,
)
from nautilus_trader.core.datetime import nanos_to_secs

_TIF = {
    'GTC': TimeInForce.GTC,
    'IOC': TimeInForce.IOC,
    'FOK': TimeInForce.FOK,
}


class PanelStrategy(Strategy):
    """
    Торговая панель Nautilus.
    Получает ордера через manual_order_queue (dict из Socket.IO).
    Отправляет события обратно через gui_queue.

    Архитектура:
        UI (index.html)
          ↓ socket.emit("submit_order", {...})
        Flask (flask_server.py)
          ↓ manual_order_queue.put(data)
        PanelStrategy.on_trade_tick → _process_order(data)
          ↓ self.submit_order(nt_order)        → Nautilus → Binance
          ↓ gui_queue.put(("order_response", ...))
        Flask queue_reader
          ↓ socketio.emit("order_response", ...)
        UI
    """

    def __init__(
        self,
        instrument_id: str,
        gui_queue,
        manual_order_queue,
    ):
        super().__init__()
        self.instrument_id     = InstrumentId.from_str(instrument_id)
        self.gui_queue         = gui_queue
        self.manual_order_queue = manual_order_queue

        self._last_price       = 0.0
        self._tick_n           = 0
        self._account_interval = 300          # отчёт о счёте каждые N тиков
        self._account_ok       = False        # True как только хотя бы раз получили реальный счёт
        self._ob_tick_n        = 0            # счётчик для throttle стакана

        # BNB — цена для расчёта дисконта комиссии
        self._bnb_price        = 0.0
        self._bnb_instrument_id = InstrumentId.from_str("BNBUSDT.BINANCE")

        # Real-ордера ожидающие OCO (SL + TP после fill)
        self._pending_exits: dict = {}        # client_order_id → (sl, tp)

        # Paper-позиция (независима от NT, только внутри стратегии)
        self._paper_pos:  dict | None = None  # {side, entry, sl, tp, qty}
        self._paper_pnl   = 0.0
        self._paper_wins  = 0
        self._paper_losses = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def on_start(self):
        self.subscribe_trade_ticks(self.instrument_id)
        # Стакан для расчёта LOB impact (slippage / price impact)
        try:
            self.subscribe_order_book_deltas(self.instrument_id)
        except Exception:
            pass
        # BNB/USDT — для расчёта реального дисконта комиссии
        self.subscribe_trade_ticks(self._bnb_instrument_id)
        self._report_account()
        self._emit("order_response", {
            "status": "ok",
            "msg": "[Panel] Подключено к Nautilus Trader",
        })

    # ── Tick ──────────────────────────────────────────────────────────────────
    def on_trade_tick(self, tick: TradeTick):
        # BNB/USDT — только обновляем курс, дальше не идём
        if tick.instrument_id == self._bnb_instrument_id:
            self._bnb_price = float(tick.price)
            return

        price = float(tick.price)
        self._last_price = price
        self._tick_n += 1

        # 1. Ручные ордера из UI
        while not self.manual_order_queue.empty():
            try:
                self._process_order(self.manual_order_queue.get_nowait())
            except Exception:
                pass

        # 2. Paper SL/TP мониторинг
        if self._paper_pos:
            mp = self._paper_pos
            if mp['side'] == 'BUY':
                if mp['tp'] > 0 and price >= mp['tp']:
                    self._close_paper(price, 'tp')
                elif mp['sl'] > 0 and price <= mp['sl']:
                    self._close_paper(price, 'sl')
            else:  # SELL
                if mp['tp'] > 0 and price <= mp['tp']:
                    self._close_paper(price, 'tp')
                elif mp['sl'] > 0 and price >= mp['sl']:
                    self._close_paper(price, 'sl')

        # 3. Обновление стакана в UI (каждые 5 тиков)
        self._ob_tick_n += 1
        if self._ob_tick_n % 5 == 0:
            self._emit_order_book()

        # 4. Периодический отчёт о счёте (real)
        if self._tick_n % self._account_interval == 0:
            self._report_account()

    def _get_val(self, obj, attr):
        """Вспомогательная функция для извлечения значения (свойство или метод)."""
        val = getattr(obj, attr, None)
        if callable(val):
            return val()
        return val

    # ── Разбор входящего ордера ───────────────────────────────────────────────
    def _process_order(self, data: dict):
        # Специальная команда — отмена всех ордеров
        if data.get('action') == 'cancel_all':
            self.cancel_all_manual()
            return

        # Обновление SL/TP открытой paper-позиции из UI
        if data.get('action') == 'update_sltp':
            new_sl = float(data.get('sl') or 0)
            new_tp = float(data.get('tp') or 0)
            if self._paper_pos:
                if new_sl > 0:
                    self._paper_pos['sl'] = new_sl
                if new_tp > 0:
                    self._paper_pos['tp'] = new_tp
                self._emit("order_response", {
                    "status": "ok",
                    "msg": (f"[PAPER] SL/TP обновлён: "
                            f"SL={self._paper_pos['sl']:.4f}  "
                            f"TP={self._paper_pos['tp']:.4f}"),
                })
            return

        mode        = data.get('mode', 'paper')
        side_str    = data.get('side', 'BUY')
        otype       = (data.get('order_type') or 'LIMIT').upper().replace(' ', '_')
        price       = float(data.get('price') or self._last_price or 0)
        qty         = float(data.get('qty') or 0.01)
        tif_str     = (data.get('tif') or 'GTC').upper()
        reduce_only = bool(data.get('reduce_only'))
        post_only   = bool(data.get('post_only'))
        sl_en       = bool(data.get('sl_enabled'))
        tp_en       = bool(data.get('tp_enabled'))
        sl_val      = float(data.get('sl_value') or 0)
        tp_val      = float(data.get('tp_value') or 0)

        instrument = self.cache.instrument(self.instrument_id)
        if not instrument:
            self._emit("order_response", {
                "status": "error",
                "msg": "Инструмент не загружен — подождите подключения",
            })
            return

        side = OrderSide.BUY if side_str == 'BUY' else OrderSide.SELL
        tif  = _TIF.get(tif_str, TimeInForce.GTC)

        # ── LOB impact: VWAP, Slippage, Price Impact ─────────────────────────
        lob       = self._calculate_lob_impact(side_str, qty)
        lob_vwap  = lob["vwap"] if lob["vwap"] > 0 else price
        exec_price = lob_vwap  # цена для расчёта номинала

        # ── Динамические комиссии из NT инструмента ──────────────────────────
        tick_size  = float(instrument.price_increment)
        maker_rate = float(getattr(instrument, 'maker_fee', Decimal('0.001')))
        taker_rate = float(getattr(instrument, 'taker_fee', Decimal('0.001')))
        is_maker   = post_only or otype in ('LIMIT', 'STOP_LIMIT', 'LIT', 'LIMIT_IF_TOUCHED')
        fee_rate   = maker_rate if is_maker else taker_rate
        notional   = exec_price * qty
        fee_usdt   = notional * fee_rate

        # ── BNB дисконт 25% — в USDT и в BNB ───────────────────────────────
        fee_bnb_usdt   = fee_usdt * 0.75   # стоимость комиссии при оплате BNB
        fee_bnb_amount = (fee_bnb_usdt / self._bnb_price
                          if self._bnb_price > 0 else 0.0)

        # Отправляем оценку в UI сразу (до исполнения)
        self._emit("commission_update", {
            "notional":       round(notional, 4),
            "maker_fee_usdt": round(notional * maker_rate, 4),
            "taker_fee_usdt": round(notional * taker_rate, 4),
            "fee_usdt":       round(fee_usdt, 4),
            "fee_bnb_usdt":   round(fee_bnb_usdt, 4),
            "fee_bnb_amount": round(fee_bnb_amount, 6),
            "bnb_price":      round(self._bnb_price, 2),
            "lob_vwap":       round(lob_vwap, 6),
            "slippage_usdt":  round(lob["slippage_usdt"], 4),
            "impact_pct":     round(lob["impact_pct"], 4),
            "depth_ok":       lob["depth_ok"],
            "tick_size":      tick_size,
        })

        sl = sl_val if sl_en else 0
        tp = tp_val if tp_en else 0

        if mode == 'paper':
            self._execute_paper(side_str, otype, price, qty, sl, tp, fee_usdt)
        elif mode == 'real':
            self._execute_real(
                side, otype, price, qty, tif,
                post_only, reduce_only, sl, tp, instrument,
            )
        else:
            self._emit("order_response", {"status": "error", "msg": f"Неизвестный режим: {mode}"})

    # ── Paper-исполнение ─────────────────────────────────────────────────────
    def _execute_paper(self, side_str, otype, price, qty, sl, tp, fee_usdt):
        if self._paper_pos:
            self._emit("order_response", {
                "status": "error",
                "msg": "Уже есть открытая paper-позиция",
            })
            return

        fill = self._last_price if otype in ('MARKET', 'MIT', 'STOP_MARKET', 'MARKET_IF_TOUCHED') else price
        rr   = round((tp - fill) / max(abs(fill - sl), 1e-9), 2) if sl and tp else 0
        self._paper_pos = {'side': side_str, 'entry': fill, 'sl': sl, 'tp': tp, 'qty': qty}

        self._emit("signal", {
            "action": "enter", "price": fill,
            "sl": sl, "tp": tp, "rr": rr, "proba": 0.0,
        })
        self._emit("order_response", {
            "status": "ok",
            "msg": (f"[PAPER] {side_str} {qty:.4f}@{fill:.4f}"
                    f"  SL={sl:.4f}  TP={tp:.4f}"
                    f"  R:R={rr:.2f}  fee≈${fee_usdt:.4f}"),
        })

    def _close_paper(self, exit_price: float, reason: str):
        mp = self._paper_pos
        self._paper_pos = None
        entry = mp['entry']
        pnl   = (exit_price - entry) / max(entry, 1e-9) * mp['qty'] * entry
        if reason == 'tp':
            self._paper_wins   += 1
        else:
            self._paper_losses += 1
        self._paper_pnl += pnl

        self._emit("signal", {
            "action": "exit", "reason": reason,
            "price": exit_price,
            "pnl": round(pnl, 2),
            "total_pnl": round(self._paper_pnl, 2),
        })
        self._emit("order_response", {
            "status": "ok",
            "msg": (f"[PAPER] Exit {reason.upper()} @{exit_price:.4f}"
                    f"  PnL={'+'if pnl>=0 else''}{pnl:.2f}$"
                    f"  Total={'+'if self._paper_pnl>=0 else''}{self._paper_pnl:.2f}$"),
        })

    # ── Real-исполнение (все типы ордеров NT) ─────────────────────────────────
    def _execute_real(self, side, otype, price, qty, tif,
                      post_only, reduce_only, sl, tp, instrument):
        try:
            qty_obj   = instrument.make_qty(qty)
            price_obj = instrument.make_price(price)
            tick      = float(instrument.price_increment)

            if otype == 'MARKET':
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    time_in_force=TimeInForce.IOC,
                    reduce_only=reduce_only,
                )
            elif otype in ('LIMIT', 'MARKETABLE_LIMIT'):
                order = self.order_factory.limit(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    price=price_obj,
                    time_in_force=tif,
                    post_only=post_only,
                    reduce_only=reduce_only,
                )
            elif otype == 'STOP_MARKET':
                order = self.order_factory.stop_market(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    trigger_price=price_obj,
                    trigger_type=TriggerType.LAST_PRICE,
                    reduce_only=reduce_only,
                )
            elif otype == 'STOP_LIMIT':
                # trigger чуть выше цены (BUY) или ниже (SELL)
                trigger = price + tick if side == OrderSide.BUY else price - tick
                order = self.order_factory.stop_limit(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    price=price_obj,
                    trigger_price=instrument.make_price(trigger),
                    time_in_force=tif,
                    post_only=post_only,
                    reduce_only=reduce_only,
                )
            elif otype in ('MIT', 'MARKET_IF_TOUCHED'):
                order = self.order_factory.market_if_touched(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    trigger_price=price_obj,
                    trigger_type=TriggerType.LAST_PRICE,
                    reduce_only=reduce_only,
                )
            elif otype in ('LIT', 'LIMIT_IF_TOUCHED'):
                trigger = price + tick if side == OrderSide.BUY else price - tick
                order = self.order_factory.limit_if_touched(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    price=price_obj,
                    trigger_price=instrument.make_price(trigger),
                    trigger_type=TriggerType.LAST_PRICE,
                    time_in_force=tif,
                    reduce_only=reduce_only,
                )
            else:
                # Fallback: Limit
                order = self.order_factory.limit(
                    instrument_id=self.instrument_id,
                    order_side=side,
                    quantity=qty_obj,
                    price=price_obj,
                    time_in_force=tif,
                )

            # Сохраняем SL/TP — выставим как OCO после fill
            if sl > 0 or tp > 0:
                self._pending_exits[order.client_order_id] = (sl, tp)

            self.submit_order(order)
            self._emit("order_response", {
                "status": "ok",
                "msg": f"[REAL] SUBMITTED {side} {qty:.4f}@{price:.4f} [{otype}]",
            })

        except Exception as e:
            self._emit("order_response", {
                "status": "error",
                "msg": f"[REAL] Ошибка отправки: {e}",
            })

    # ── Order events ──────────────────────────────────────────────────────────
    def on_order_filled(self, event: OrderFilled):
        cid      = event.client_order_id
        fill_px  = float(event.last_px)
        fill_qty = event.last_qty
        instrument = self.cache.instrument(self.instrument_id)

        # OCO SL + TP после исполнения ручного ордера
        if cid in self._pending_exits and instrument:
            sl, tp = self._pending_exits.pop(cid)
            if sl > 0:
                sl_order = self.order_factory.stop_market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=fill_qty,
                    trigger_price=instrument.make_price(sl),
                    trigger_type=TriggerType.LAST_PRICE,
                    reduce_only=True,
                )
                self.submit_order(sl_order)
            if tp > 0:
                tp_order = self.order_factory.limit(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=fill_qty,
                    price=instrument.make_price(tp),
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )
                self.submit_order(tp_order)
            self._emit("signal", {
                "action": "enter", "price": fill_px,
                "sl": sl, "tp": tp,
                "rr": round((tp - fill_px) / max(abs(fill_px - sl), 1e-9), 2) if sl and tp else 0,
                "proba": 0.0,
            })

        self._emit("order_response", {
            "status": "ok",
            "msg": f"[FILLED] {float(fill_qty):.4f} @ {fill_px:.4f}",
        })
        self._report_account()

    def on_order_canceled(self, event: OrderCanceled):
        self._emit("order_response", {
            "status": "ok",
            "msg": f"[CANCELLED] {event.client_order_id}",
        })

    def on_order_rejected(self, event: OrderRejected):
        self._emit("order_response", {
            "status": "error",
            "msg": f"[REJECTED] {event.reason}",
        })

    # ── Cancel All ────────────────────────────────────────────────────────────
    def cancel_all_manual(self):
        open_orders = self.cache.orders_open(instrument_id=self.instrument_id)
        for o in open_orders:
            self.cancel_order(o)
        self._emit("cancel_response", {
            "msg": f"Отменено {len(open_orders)} ордер(ов)",
        })

    # ── LOB Impact ───────────────────────────────────────────────────────────
    def _calculate_lob_impact(self, side: str, qty: float) -> dict:
        """
        Проходит по уровням стакана для объёма qty.
        Возвращает: vwap, slippage_usdt, impact_pct, depth_ok.
        Если стакан ещё не заполнен — fallback: 2 tick_size.
        """
        # Tick-size fallback (когда стакан ещё не готов)
        instrument = self.cache.instrument(self.instrument_id)
        tick_size  = float(instrument.price_increment) if instrument else 0.0
        best       = self._last_price if self._last_price > 0 else 1.0
        # 2 тика = консервативная оценка рыночного ордера
        fallback_slip   = 2 * tick_size * qty
        fallback_impact = fallback_slip / (best * qty) * 100 if best > 0 and qty > 0 else 0.0
        _empty = {
            "vwap":          best,
            "slippage_usdt": round(fallback_slip,   4),
            "impact_pct":    round(fallback_impact,  4),
            "depth_ok":      False,   # depth_ok=False сигнализирует UI: это estimate
        }
        try:
            ob = self.cache.order_book(self.instrument_id)
            if ob is None:
                return _empty
            
            # В новых версиях asks() и bids() — это методы
            levels = list(ob.asks()) if side == 'BUY' else list(ob.bids())
            if not levels:
                return _empty

            best_price = float(self._get_val(levels[0], 'price'))
            remaining  = qty
            total_cost = 0.0

            for lvl in levels:
                lp   = float(self._get_val(lvl, 'price'))
                lq   = float(self._get_val(lvl, 'size'))
                take = min(remaining, lq)
                total_cost += take * lp
                remaining  -= take
                if remaining <= 0:
                    break

            depth_ok = remaining <= 0
            if remaining > 0:
                # Стакан не покрыл объём — добиваем по последней цене
                total_cost += remaining * float(self._get_val(levels[-1], 'price'))

            vwap          = total_cost / qty if qty > 0 else best_price
            slippage_usdt = abs(vwap - best_price) * qty
            impact_pct    = (abs(vwap - best_price) / best_price * 100
                             if best_price else 0.0)
            return {
                "vwap":          round(vwap, 6),
                "slippage_usdt": round(slippage_usdt, 4),
                "impact_pct":    round(impact_pct, 4),
                "depth_ok":      depth_ok,
            }
        except Exception:
            return _empty

    # ── Order Book → Browser ─────────────────────────────────────────────────
    def _emit_order_book(self):
        """
        Читает top-4 ask/bid из кэша Nautilus и отправляет в браузер.
        Вызывается каждые 5 тиков — достаточно для живого отображения
        без перегрузки очереди.
        """
        try:
            ob = self.cache.order_book(self.instrument_id)
            if ob is None:
                return

            asks = list(ob.asks())   # уже отсортированы
            bids = list(ob.bids())

            if not asks and not bids:
                return

            def fmt_levels(levels, n=4):
                result = []
                for lvl in levels[:n]:
                    result.append({
                        "price": float(self._get_val(lvl, 'price')),
                        "size":  round(float(self._get_val(lvl, 'size')), 4),
                    })
                return result

            ask_levels = fmt_levels(asks, 4)
            bid_levels = fmt_levels(bids, 4)

            spread = (float(self._get_val(asks[0], 'price')) - float(self._get_val(bids[0], 'price'))
                      if asks and bids else 0.0)

            self._emit("order_book", {
                "asks":   ask_levels,
                "bids":   bid_levels,
                "spread": round(spread, 8),
            })
        except Exception:
            pass

    # ── Real balance report ───────────────────────────────────────────────────
    def _report_account(self):
        try:
            venue = Venue("BINANCE")

            # Проверяем список счётов через accounts() — НЕ вызывает Nautilus ERROR-лог.
            # portfolio.account(venue) логирует ERROR если счёт не зарегистрирован
            # (типично для SPOT+testnet без реальных credentials).
            # Предварительная проверка список избегает шумных ошибок в логах.
            try:
                all_accs = self.portfolio.accounts()
                if not all_accs:
                    return  # счёт ещё не зарегистрирован
            except Exception:
                return

            account = self.portfolio.account(venue)
            if account:
                from nautilus_trader.model.currencies import USDT
                bal_total = account.balance_total(USDT)
                bal_free  = account.balance_free(USDT)

                margins_init   = 0.0
                unrealized_pnl = 0.0
                try:
                    mi = account.margins_init(USDT)
                    if mi:
                        margins_init = round(float(mi.as_double()), 2)
                except Exception:
                    pass
                try:
                    upnl = self.portfolio.unrealized_pnl(self.instrument_id)
                    if upnl:
                        unrealized_pnl = round(float(upnl.as_double()), 2)
                except Exception:
                    pass

                self._emit("account_update", {
                    "balance_total":  round(float(bal_total.as_double()), 2) if bal_total else 0.0,
                    "balance_free":   round(float(bal_free.as_double()), 2)  if bal_free  else 0.0,
                    "margins_init":   margins_init,
                    "unrealized_pnl": unrealized_pnl,
                })
        except Exception:
            pass

    # ── Helper ───────────────────────────────────────────────────────────────
    def _emit(self, event_type: str, payload: dict):
        if self.gui_queue:
            self.gui_queue.put((event_type, payload))
