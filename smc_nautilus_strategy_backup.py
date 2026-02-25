"""
Multi-SMC Nautilus Strategy — Multi-Instrument Paper Trading.
Поддержка 30+ инструментов одновременно.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import deque
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import TradeTick as Trade
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import AggressorSide, OrderSide
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.core.datetime import nanos_to_secs
from live_packet_builder import VolumePacketBuilder, Packet, PartialPacket
from features import calculate_er
from smc_detector import detect_ote_signal
import os
import datetime


class MultiSMCStrategy(Strategy):
    def __init__(self, strategy_config: dict, symbols_configs: dict, gui_queue=None, manual_order_queue=None):
        """
        symbols_configs: { instrument_id: { 'bucket_usd': float, ... } }
        """
        super().__init__()
        self.strategy_config = strategy_config
        self.model_threshold = strategy_config.get("smc_threshold", 0.5)
        self.swing_length    = strategy_config.get("swing_length", 50)
        self.lag             = strategy_config.get("lag", 50)
        self.trade_size      = strategy_config.get("trade_size", 0.05)

        self.gui_queue = gui_queue
        self.manual_order_queue = manual_order_queue

        # Реестр состояний для каждого инструмента
        self.instruments_data = {}
        for inst_id_str, cfg in symbols_configs.items():
            inst_id = InstrumentId.from_str(inst_id_str)
            pb = cfg['packet_builder']
            hist_packets = cfg.get('hist_packets', [])
            
            q_packets = deque(maxlen=5000)
            q_closes = deque(maxlen=50)
            
            for p in hist_packets:
                q_packets.append(p)
                q_closes.append(p.close)
                
            self.instruments_data[inst_id] = {
                'packet_builder': pb,
                'packets': q_packets,
                'close_prices': q_closes,
                'pending_signal': None,
                'in_position': False,
                'entry_price_val': 0.0,
                'pending_exits': {}, # client_id -> (sl, tp)
                'active_sl_id': None,
                'active_tp_id': None,
                'warmup': self.swing_length * 4,
                'paper_pnl': 0.0,
                'paper_wins': 0,
                'paper_losses': 0,
                'last_model_proba': None
            }

        self.model = xgb.XGBClassifier()
        self.model.load_model(self.strategy_config["model_path"])

    def on_start(self):
        for inst_id, data in self.instruments_data.items():
            self.subscribe_trade_ticks(inst_id)
            
            if len(data['packets']) >= data['warmup']:
                df_history = self._to_df(inst_id)
                history_window = 500
                eval_df = df_history[-history_window:] if len(df_history) > history_window else df_history
                signal = detect_ote_signal(eval_df, swing_length=self.swing_length)
                if signal:
                    data['pending_signal'] = signal
                    print(f"[SMC][{inst_id}] Активный сигнал найден сразу после загрузки истории!")
            
    def on_update(self):
        """Проверка очереди ручных команд из Flask."""
        if self.manual_order_queue:
            import queue
            while not self.manual_order_queue.empty():
                try:
                    cmd = self.manual_order_queue.get_nowait()
                    self._handle_manual_cmd(cmd)
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[SMC] Manual cmd error: {e}")

    def _handle_manual_cmd(self, cmd: dict):
        action = cmd.get("action")
        symbol = cmd.get("symbol")
        if not symbol: return
        
        inst_id = InstrumentId.from_str(f"{symbol}.BINANCE")
        data = self.instruments_data.get(inst_id)
        if not data: return

        if action == "cancel_all":
            self.cancel_all_orders(inst_id)
            print(f"[NAUTILUS][{inst_id}] CANCEL ALL")
        elif action == "update_sltp":
            print(f"[NAUTILUS][{inst_id}] SL/TP Update via UI not implemented in Multi-mode yet")
        else:
            # Обычный submit_order из UI
            side_str = cmd.get("side", "BUY")
            side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
            price = float(cmd.get("price", 0))
            qty_val = float(cmd.get("qty", 0))
            
            instrument = self.cache.instrument(inst_id)
            if not instrument: return
            
            order = self.order_factory.limit(
                instrument_id=inst_id,
                order_side=side,
                price=instrument.make_price(price),
                quantity=instrument.make_qty(qty_val)
            )
            self.submit_order(order)
            print(f"[NAUTILUS][{inst_id}] SUBMIT MANUAL {side_str} @ {price}")
            
        # Отправляем инфо о бакетах в GUI (можно расширить для списка)
        if self.gui_queue:
            buckets = {str(i): d['packet_builder'].bucket_usd for i, d in self.instruments_data.items()}
            self.gui_queue.put(("multi_bucket_info", buckets))

    def on_trade_tick(self, trade: Trade):
        inst_id = trade.instrument_id
        if inst_id not in self.instruments_data:
            return

        data = self.instruments_data[inst_id]
        price   = float(trade.price)
        qty     = float(trade.size)
        is_sell = (trade.aggressor_side == AggressorSide.SELLER)
        ts      = trade.ts_event

        # 1. Обновляем пакет
        if data['packet_builder'] is None:
            return
        closed_packets, partial = data['packet_builder'].update(price, qty, is_sell, ts)
        
        for pkt in closed_packets:
            self._process_closed_packet(inst_id, pkt)

        # 2. Проверка сигналов и входа
        if data['pending_signal'] and not data['in_position'] and partial:
            sig = data['pending_signal']
            # Если цена ушла на перехай - инвалидируем сигнал (детектор обновит на следующем пакете)
            if price > sig['swing_high']:
                print(f"[SMC][{inst_id}] Перехай! {price:.4f} > {sig['swing_high']:.4f}. Сигнал будет обновлен.")
                data['pending_signal'] = None
            elif sig['ote_low'] <= price <= sig['ote_high']:
                # Опрашиваем модель только в зоне Sweet Spot (или OTE целиком)
                proba = self._model_proba_partial(inst_id, sig, partial)
                data['last_model_proba'] = proba
                
                # Логируем в UI периодически
                import time
                now_sec = time.time()
                if now_sec - data.get('last_log_time', 0) > 60:
                    data['last_log_time'] = now_sec
                    if self.gui_queue:
                        sym_str = str(inst_id).split(".")[0]
                        if proba >= self.model_threshold:
                            msg = f"[SMC] {sym_str} В ЗОНЕ. Модель ЗА вход ({proba*100:.0f}% \u2265 {self.model_threshold*100:.0f}%)"
                            self.gui_queue.put(("log_msg", {"msg": msg, "color": "c-g"}))
                        else:
                            msg = f"[SMC] {sym_str} В ЗОНЕ. Ждем модель ({proba*100:.0f}% < {self.model_threshold*100:.0f}%)"
                            self.gui_queue.put(("log_msg", {"msg": msg, "color": "c-y"}))

                if proba >= self.model_threshold:
                    sl = sig['stop_loss']
                    # Динамический ТП по уверенности
                    tp = sig['tp_aggressive'] if proba >= 0.75 else sig['tp_standard']
                    self._nautilus_enter(inst_id, price, sl, tp, proba)
                    data['pending_signal'] = None
            elif price < sig['ote_low'] - sig['range_size'] * 0.05:
                # Инвалидация вниз
                data['pending_signal'] = None

        if self.gui_queue and partial:
            self._send_partial(inst_id, partial)

    def _process_closed_packet(self, inst_id, packet: Packet):
        data = self.instruments_data[inst_id]
        data['packets'].append(packet)
        data['close_prices'].append(packet.close)

        warmed_up = len(data['packets']) >= data['warmup']
        signal = None
        
        if warmed_up:
            df_history = self._to_df(inst_id)
            history_window = 2000 # Достаточное окно для поиска BOS при lag=50 и swing=50
            eval_df = df_history[-history_window:] if len(df_history) > history_window else df_history
                
            signal = detect_ote_signal(eval_df, swing_length=self.swing_length)
            if signal:
                data['pending_signal'] = signal

        if self.gui_queue:
            self._send_metrics(inst_id, packet, signal)
            self.gui_queue.put(("candle", {
                "symbol": str(inst_id).split(".")[0],
                "time":  nanos_to_secs(packet.timestamp),
                "open":  packet.open,  "high": packet.high,
                "low":   packet.low,   "close": packet.close,
            }))

    def _nautilus_enter(self, inst_id, price: float, sl: float, tp: float, proba: float):
        try:
            data = self.instruments_data[inst_id]
            instrument = self.cache.instrument(inst_id)
            if not instrument: 
                print(f"[SMC_ERROR] Instrument not in cache: {inst_id}")
                return

            # Вычисляем размер поинта в монетах для сделки на ~100$ (чтобы соблюсти MIN_NOTIONAL)
            # Если price = 2$, то для 100$ нужно 50 монет
            qty_coins = max(100.0 / price, 0.01) if price > 0 else 0.01
            qty = instrument.make_qty(qty_coins)
            entry_price = instrument.make_price(price)
                
            order = self.order_factory.limit(
                instrument_id=inst_id,
                order_side=OrderSide.BUY,
                price=entry_price,
                quantity=qty,
            )
            data['pending_exits'][order.client_order_id] = (sl, tp)
            self.submit_order(order)
            print(f"[NAUTILUS][{inst_id}] SUBMIT ENTRY @ {price:.4f} sl={sl:.4f} tp={tp:.4f} proba={proba:.1%}")
        except Exception as e:
            print(f"[SMC_CRITICAL] Failed to execute enter for {inst_id}: {e}")

    def on_order_filled(self, event: OrderFilled):
        client_id = event.client_order_id
        inst_id = event.instrument_id
        data = self.instruments_data.get(inst_id)
        if not data: return
        
        if client_id in data['pending_exits']:
            data['in_position'] = True
            sl_price, tp_price = data['pending_exits'].pop(client_id)
            qty = event.last_qty
            data['entry_price_val'] = float(event.last_px)
            instrument = self.cache.instrument(inst_id)
            
            sl_price_obj = instrument.make_price(sl_price)
            tp_price_obj = instrument.make_price(tp_price)
                
            sl_order = self.order_factory.stop_market(inst_id, OrderSide.SELL, qty, trigger_price=sl_price_obj)
            tp_order = self.order_factory.limit(inst_id, OrderSide.SELL, qty, price=tp_price_obj)
            
            self.submit_order(sl_order)
            self.submit_order(tp_order)
            data['active_sl_id'] = sl_order.client_order_id
            data['active_tp_id'] = tp_order.client_order_id
            
            print(f"[NAUTILUS][{inst_id}] ENTRY FILLED @ {data['entry_price_val']:.4f}. Placed SL={sl_price:.4f}, TP={tp_price:.4f}")
            
            # --- Логируем баланс после Входа ---
            portfolio = self.portfolio
            if portfolio:
                acc = portfolio.account(self.portfolio.account_ids[0])
                usdt_bal = acc.balance(Currency.from_str("USDT"))
                btc_bal = acc.balance(Currency.from_str("BTC"))
                print(f"[TEST] БАЛАНС ПОСЛЕ ВХОДА: {usdt_bal.free.as_double():.2f} USDT | {btc_bal.free.as_double():.5f} BTC")
            # ------------------------------------
            
            # Логируем вход в CSV
            try:
                with open("trades_history.csv", "a", encoding="utf-8") as f:
                    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{dt_str},{inst_id},ENTER,{data['entry_price_val']:.4f},{sl_price:.4f},{tp_price:.4f},0.0\n")
            except: pass

            if self.gui_queue:
                self.gui_queue.put(("signal", {
                    "symbol": str(inst_id).split(".")[0],
                    "price": data['entry_price_val'],
                    "sl": round(sl_price, 6), "tp": round(tp_price, 6),
                    "action": "enter"
                }))

        elif client_id in (data['active_sl_id'], data['active_tp_id']):
            is_tp = (client_id == data['active_tp_id'])
            reason = "tp" if is_tp else "sl"
            other_id = data['active_sl_id'] if is_tp else data['active_tp_id']
            
            if is_tp: data['paper_wins'] += 1
            else: data['paper_losses'] += 1
                
            if other_id:
                other_order = self.cache.order(other_id)
                if other_order: self.cancel_order(other_order)
                    
            data['active_sl_id'] = data['active_tp_id'] = None
            data['in_position'] = False
            
            pnl_usd = (float(event.last_px) - data['entry_price_val']) / data['entry_price_val'] * 100.0
            if not is_tp: pnl_usd = -abs(pnl_usd) # loss

            data['paper_pnl'] += pnl_usd
            print(f"[NAUTILUS][{inst_id}] POSITION CLOSED by {reason.upper()} @ {float(event.last_px):.4f}. PnL: {pnl_usd:.2f}%")
            
            # --- Логируем баланс после Выхода ---
            portfolio = self.portfolio
            if portfolio:
                acc = portfolio.account(self.portfolio.account_ids[0])
                usdt_bal = acc.balance(Currency.from_str("USDT"))
                btc_bal = acc.balance(Currency.from_str("BTC"))
                print(f"[TEST] БАЛАНС ПОСЛЕ ВЫХОДА: {usdt_bal.free.as_double():.2f} USDT | {btc_bal.free.as_double():.5f} BTC")
            # ------------------------------------
            
            # Логируем выход в CSV
            try:
                with open("trades_history.csv", "a", encoding="utf-8") as f:
                    dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{dt_str},{inst_id},EXIT_{reason.upper()},{float(event.last_px):.4f},,,{pnl_usd:.2f}\n")
            except: pass

            if self.gui_queue:
                self.gui_queue.put(("signal", {
                    "symbol": str(inst_id).split(".")[0],
                    "price": float(event.last_px),
                    "action": "exit", "reason": reason,
                    "pnl": round(pnl_usd, 2),
                    "total_pnl": round(data['paper_pnl'], 2),
                }))

    def _model_proba_partial(self, inst_id, signal: dict, partial: PartialPacket) -> float:
        try:
            data = self.instruments_data[inst_id]
            r = signal['range_size']
            cur = partial.close
            ote = (cur - signal['ote_low']) / r if r > 0 else 0.0
            
            prices = getattr(partial, 'prices', [])
            volatility = float(np.std(np.diff(np.log(prices)))) if len(prices) >= 2 else 0.0

            feat = pd.DataFrame([{
                "ote_position":    ote,
                "retracement_pct": (signal['bos_high'] - cur) / signal['bos_high'] * 100,
                "time_since_bos":  len(data['packets']) - signal['bos_index'],
                "range_size":      r,
                "vpin":            partial.vpin,
                "volume":          getattr(partial, 'volume_usd', 0.0),
                "volatility":      volatility,
                "imbalance":       partial.imbalance,
                "candle_range":    (partial.high - partial.low) / partial.open * 100 if partial.open > 0 else 0.0,
                "trades_count":    partial.trades_count,
            }])
            return float(self.model.predict_proba(feat)[0][1])
        except: return 0.0

    def on_order_canceled(self, event):
        print(f"[NAUTILUS][{event.instrument_id}] ORDER CANCELED: {event.client_order_id}")

    def on_order_rejected(self, event):
        print(f"[NAUTILUS][{event.instrument_id}] ORDER REJECTED: {event.client_order_id} - {event.reason}")

    def on_order_accepted(self, event):
        print(f"[NAUTILUS][{event.instrument_id}] ORDER ACCEPTED: {event.client_order_id}")

    def _send_metrics(self, inst_id, packet: Packet, signal: dict | None):
        data = self.instruments_data[inst_id]
        er = calculate_er(list(data['close_prices']), period=20)

        current_signal = data['pending_signal'] or signal
        ote_dist, retr = None, None
        price = packet.close

        if current_signal:
            ote_high = current_signal['ote_high']
            ote_low = current_signal['ote_low']
            # Правильный расчёт расстояния до OTE зоны
            if price > ote_high:
                # Цена ВЫШЕ зоны — нужно упасть, расстояние положительное
                ote_dist = (price - ote_high) / ote_high * 100
            elif price < ote_low:
                # Цена НИЖЕ зоны — ушла за invalidation
                ote_dist = (ote_low - price) / ote_low * -100
            else:
                # Цена В ЗОНЕ
                ote_dist = 0.0
            retr = (current_signal['bos_high'] - price) / max(current_signal['bos_high'], 1e-9) * 100

        wp = round(len(data['packets']) / data['warmup'] * 100, 1) if len(data['packets']) < data['warmup'] else 100.0

        res = {
            "symbol": str(inst_id).split(".")[0],
            "imbalance": round(packet.imbalance * 100, 2),
            "vpin": round(packet.vpin * 100, 2),
            "er": round(er * 100, 1),
            "volatility": round(packet.candle_range * 100, 4),
            "bucket": int(data['packet_builder'].bucket_usd),
            "ote_distance": round(ote_dist, 2) if ote_dist is not None else None,
            "retracement_pct": round(retr, 2) if retr is not None else None,
            "warmup_pct": wp,
            "packets_count": len(data['packets']),
            "paper_pnl": round(data['paper_pnl'], 2),
            "in_position": data['in_position'],
            "has_signal": current_signal is not None,
            "model_proba": data.get('last_model_proba')
        }
        if current_signal:
            # Entry 0.705 Sweet Spot (лимитка)
            res["entry_705"] = round(current_signal['entry_price'], 6)
            # Зона OTE (границы)
            res["ote_high"] = round(current_signal['ote_high'], 6)
            res["ote_low"] = round(current_signal['ote_low'], 6)
            # Структурные уровни
            res["structural_sl"] = round(current_signal['stop_loss'], 6)
            res["structural_tp"] = round(current_signal['tp_standard'], 6)
            res["tp_conservative"] = round(current_signal['tp_conservative'], 6)
            res["tp_aggressive"] = round(current_signal['tp_aggressive'], 6)
            res["swing_high"] = round(current_signal['swing_high'], 6)
            res["swing_low"] = round(current_signal['swing_low'], 6)
            res["range_size"] = round(current_signal['range_size'], 6)
            # R:R
            risk = abs(current_signal['entry_price'] - current_signal['stop_loss'])
            reward = abs(current_signal['tp_standard'] - current_signal['entry_price'])
            res["rr_ratio"] = round(reward / risk, 1) if risk > 0 else 0
            
        self.gui_queue.put(("metrics", res))

    def _send_partial(self, inst_id, partial: PartialPacket):
        data = self.instruments_data[inst_id]
        self.gui_queue.put(("partial", {
            "symbol": str(inst_id).split(".")[0],
            "time": nanos_to_secs(partial.timestamp),
            "close": partial.close,
            "imbalance": round(partial.imbalance * 100, 2),
            "fill_pct": round(partial.fill_pct * 100, 1)
        }))

    def _to_df(self, inst_id) -> pd.DataFrame:
        data = self.instruments_data[inst_id]
        return pd.DataFrame([{
            "timestamp": p.timestamp, "open": p.open, "high": p.high,
            "low": p.low, "close": p.close, "volume": p.volume,
            "buy_volume": p.buy_volume, "sell_volume": p.sell_volume,
            "trades_count": p.trades_count,
        } for p in data['packets']])
