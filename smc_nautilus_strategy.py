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
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.enums import AggressorSide, OrderSide
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.currencies import Currency, USDT
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.core.datetime import nanos_to_secs
from decimal import Decimal
from live_packet_builder import VolumePacketBuilder, Packet, PartialPacket
from features import calculate_er
from smc_detector import detect_ote_signal
import os
import datetime


class MultiSMCStrategy(Strategy):
    def __init__(self, strategy_config: dict, symbols_configs: dict, gui_queue=None, manual_order_queue=None, sandbox_client_ref=None):
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
        self._sandbox_client_ref = sandbox_client_ref  # [SandboxExecutionClient] for hot-insert

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
                'dead_signals': set(), # Хранит swing_low отмененных зон, чтобы не воскрешать их
                'in_position': False,
                'entry_price_val': 0.0,
                'pending_entries': {}, # client_order_id -> (signal, sl, tp) 
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
            # ... rest of on_start logic ...

            if len(data['packets']) >= data['warmup']:
                df_history = self._to_df(inst_id)
                history_window = 2000
                eval_df = df_history[-history_window:] if len(df_history) > history_window else df_history
                signal = detect_ote_signal(eval_df, swing_length=self.swing_length)
                if signal:
                    # ⚠️ ПРОВЕРКА: если свеча закрылась ниже swing_low — зона сломана, не входим
                    # Прострел до 5% внутри свечи = норма (SMC liquidity grab)
                    cur_price = float(data['packets'][-1].close)
                    last_closed = cur_price # В истории последняя свеча закрыта
                    invalid_reason = self._check_signal_invalid(signal, cur_price, last_closed)
                    if invalid_reason:
                        print(f"[SMC][{inst_id}] Сигнал из истории ИНВАЛИДЕН: {invalid_reason}. Пропускаем.")
                        if self.gui_queue:
                            sym_str = str(inst_id).split('.')[0]
                            self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} зона инвалидна: {invalid_reason}", "color": "c-r"}))
                        continue

                    data['pending_signal'] = signal
                    print(f"[SMC][{inst_id}] Активный сигнал найден сразу после загрузки истории!")

                    proba = self._model_proba_partial(inst_id, signal, data['packets'][-1])
                    data['last_model_proba'] = proba

                    if proba >= self.model_threshold:
                        sl = signal['stop_loss']
                        tp = signal['tp_aggressive'] if proba >= 0.75 else signal['tp_standard']
                        success = self._nautilus_enter_limit_advance(inst_id, signal['entry_price'], sl, tp, signal)
                        if success:
                            data['pending_signal'] = None


            # Отправка первичных метрик после загрузки истории, чтобы убрать статус "WARMUP 0%" в UI
            if self.gui_queue and len(data['packets']) > 0:
                self._send_metrics(inst_id, data['packets'][-1], data['pending_signal'])
            
    def on_instrument(self, instrument):
        inst_id = instrument.id
        print(f"[SMC DEBUG] Инструмент {inst_id} загружен в кэш.")
        if inst_id in self.instruments_data:
            data = self.instruments_data[inst_id]
            sig = data.get('pending_signal')
            # Если сигнал висит с момента on_start, ставим лимитку сейчас
            if sig and not data['in_position'] and len(data['pending_entries']) == 0:
                # Повторная проверка валидности — между on_start и on_instrument цена могла измениться
                cur_price = float(data['packets'][-1].close) if data['packets'] else 0
                last_closed = cur_price
                invalid_reason = self._check_signal_invalid(sig, cur_price, last_closed)
                if invalid_reason:
                    print(f"[SMC][{inst_id}] Отложенный сигнал ИНВАЛИДЕН: {invalid_reason}. Отменяем.")
                    data['pending_signal'] = None
                    if self.gui_queue:
                        sym_str = str(inst_id).split('.')[0]
                        self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} зона инвалидна: {invalid_reason}", "color": "c-r"}))
                    return
                proba = data.get('last_model_proba')
                if proba is None:
                    proba = self._model_proba_partial(inst_id, sig, data['packets'][-1])
                    data['last_model_proba'] = proba
                if proba >= self.model_threshold:
                    sl = sig['stop_loss']
                    tp = sig['tp_aggressive'] if proba >= 0.75 else sig['tp_standard']
                    success = self._nautilus_enter_limit_advance(inst_id, sig['entry_price'], sl, tp, sig)
                    if success:
                        data['pending_signal'] = None

    def on_update(self):
        # BUG 8 FIX: Дублирование убрано. Очередь проверяется в on_trade_tick → _check_queue.
        # on_update вызывается Nautilus периодически — используем только как fallback.
        self._check_queue()

    def _check_queue(self):
        if not self.manual_order_queue:
            return
        import queue
        while not self.manual_order_queue.empty():
            try:
                cmd = self.manual_order_queue.get_nowait()
                print(f"[QUEUE] Got command: action={cmd.get('action')}, symbol={cmd.get('symbol','?')}")
                self._handle_manual_cmd(cmd)
            except queue.Empty:
                break
            except Exception as e:
                print(f"[SMC] Ошибка обработки очереди: {e}")
                import traceback
                traceback.print_exc()

    def _handle_manual_cmd(self, cmd: dict):
        action = cmd.get("action")
        
        # Инструмент уже скачан во Flask, просто загружаем в оперативную память
        if action == "add_instrument_ready":
            sym = cmd.get("symbol")
            inst_id = InstrumentId.from_str(f"{sym}.BINANCE")
            hist_packets = cmd.get("hist_packets", [])
            pb = cmd.get("packet_builder")
            
            from collections import deque
            packet_window = getattr(self, "packet_window", 5000)
            
            q_packets = deque(maxlen=packet_window)
            q_closes = deque(maxlen=self.lag)
            
            for p in hist_packets:
                q_packets.append(p)
                q_closes.append(p.close)
            
            self.instruments_data[inst_id] = {
                'packet_builder': pb,
                'packets': q_packets,
                'close_prices': q_closes,
                'pending_signal': None,
                'pending_entries': {},
                'pending_exits': {},       # BUG 6 FIX: было пропущено
                'active_sl_id': None,
                'active_tp_id': None,
                'in_position': False,
                'entry_price_val': 0.0,
                'paper_pnl': 0.0,
                'paper_wins': 0,
                'paper_losses': 0,
                'last_model_proba': None,
                'dead_signals': set(),
                'warmup': self.swing_length * 4
            }
            
            # --- БЕЗОПАСНЫЙ HOT-INSERT АКТИВА ---
            # С load_all=True Nautilus сам загрузил ~428 USDT пар с правильными параметрами.
            # 13 монет (PEPE, SHIB и др.) не загрузились из-за maxQty > uint64 limit.
            print(f"[HOT_INSERT] Checking cache for {inst_id}...")
            cached_inst = self.cache.instrument(inst_id)
            if not cached_inst:
                exchange_info = cmd.get("exchange_info")
                if exchange_info:
                    cached_inst = self._register_instrument_from_exchange_info(inst_id, exchange_info)
                if not cached_inst:
                    print(f"[HOT_INSERT] WARN: {inst_id} не удалось создать. Пропускаем.")
                    return
            print(f"[HOT_INSERT] OK: {inst_id} в кэше (base={cached_inst.base_currency}, price_prec={cached_inst.price_precision})")

            print(f"[MULTI-SMC] Подписка на {sym} (инструмент уже в кэше)")
            self.subscribe_trade_ticks(inst_id)
            print(f"[HOT_INSERT] subscribe_trade_ticks({inst_id}) OK")
            
            # Сразу обновляем интерфейс и проверяем зону SMC
            if len(hist_packets) > 0:
                if self.gui_queue:
                    self._send_metrics(inst_id, hist_packets[-1], None)
                
                if len(hist_packets) >= self.instruments_data[inst_id]['warmup']:
                    from smc_detector import detect_ote_signal
                    df_history = self._to_df(inst_id)
                    eval_df = df_history[-2000:] if len(df_history) > 2000 else df_history
                    signal = detect_ote_signal(eval_df, swing_length=self.swing_length)
                    
                    if signal:
                        cur_price = float(hist_packets[-1].close)
                        invalid_reason = self._check_signal_invalid(signal, cur_price, cur_price)
                        if not invalid_reason:
                            self.instruments_data[inst_id]['pending_signal'] = signal
                            proba = self._model_proba_partial(inst_id, signal, hist_packets[-1])
                            self.instruments_data[inst_id]['last_model_proba'] = proba
                            if proba >= self.model_threshold:
                                sl = signal['stop_loss']
                                tp = signal['tp_aggressive'] if proba >= 0.75 else signal['tp_standard']
                                self._nautilus_enter_limit_advance(inst_id, signal['entry_price'], sl, tp, signal)
                            else:
                                if self.gui_queue:
                                    m = f"[SMC] {sym} зона ({int(proba*100)}%) ниже порога {int(self.model_threshold*100)}%"
                                    self.gui_queue.put(("log_msg", {"msg": m, "color": "c-w"}))
                        else:
                            if self.gui_queue:
                                m = f"[SMC] {sym} зона сразу инвалидна: {invalid_reason}"
                                self.gui_queue.put(("log_msg", {"msg": m, "color": "c-r"}))
                                # Форсируем обновление UI, чтобы сбросить warmup
                                self._send_metrics(inst_id, hist_packets[-1], None)
                    else:
                        # Если зон нет, просто обновляем еще раз для уверенности
                        if self.gui_queue:
                            self._send_metrics(inst_id, hist_packets[-1], None)
            return

        # ---- Остальные ручные команды из UI ----
        symbol = cmd.get("symbol")
        if not symbol: return
        
        inst_id = InstrumentId.from_str(f"{symbol}.BINANCE")
        if action == "cancel_all" and inst_id in self.instruments_data:
            self.cancel_all_orders(inst_id)
            print(f"[NAUTILUS][{inst_id}] CANCEL ALL")
        elif action not in ["update_sltp"]:
            # Логика submit_order (ручной вход с UI)
            data = self.instruments_data.get(inst_id)
            if not data: return
            
            side_str = cmd.get("side", "BUY")
            side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
            price = float(cmd.get("price", 0))
            qty_val = float(cmd.get("qty", 0))
            
            instrument = self.cache.instrument(inst_id)
            if instrument:
                order = self.order_factory.limit(
                    instrument_id=inst_id, order_side=side,
                    price=instrument.make_price(price), quantity=instrument.make_qty(qty_val)
                )
                self.submit_order(order)
                print(f"[NAUTILUS][{inst_id}] SUBMIT MANUAL {side_str} @ {price}")
            
        # Отправляем инфо о бакетах в GUI (можно расширить для списка)
        if self.gui_queue:
            buckets = {str(i): d['packet_builder'].bucket_usd for i, d in self.instruments_data.items()}
            self.gui_queue.put(("multi_bucket_info", buckets))

    def on_trade_tick(self, trade: Trade):
        self._check_queue()
        
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

        # 2. Проверка сигналов и Динамическая Защита (Dynamic Cancel)
        
        # А) Проверка pending_signal на каждом живом тике
        # BUG 14 FIX: защита от пустого массива пакетов (если история не загрузилась)
        last_closed_price = float(data['packets'][-1].close) if data.get('packets') and len(data['packets']) > 0 else price
        
        if data['pending_signal'] and partial:
            sig = data['pending_signal']
            # Инвалидация 1: перехай или касание TP — движение ушло без нас
            if price > sig['swing_high'] or price >= sig['tp_standard']:
                print(f"[SMC][{inst_id}] Цель достигнута без нас! {price:.4f} >= TP/High. Сигнал инвалидирован.")
                data['dead_signals'].add(round(sig['swing_low'], 8))
                data['pending_signal'] = None
            # Инвалидация 2: свеча ЗАКРЫЛАСЬ ниже swing_low (настоящий Слом Структуры)
            elif last_closed_price < sig['swing_low']:
                print(f"[SMC][{inst_id}] СЛОМ СТРУКТУРЫ (свеча закрылась ниже {sig['swing_low']:.4f}). Сигнал инвалидирован.")
                if self.gui_queue:
                    sym_str = str(inst_id).split('.')[0]
                    self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} слом структуры (закрытие ниже swing_low)", "color": "c-r"}))
                data['dead_signals'].add(round(sig['swing_low'], 8))
                data['pending_signal'] = None
            # Инвалидация 3: слишком глубокий прострел внутри свечи (>5%)
            elif price < sig['swing_low'] * 0.95:
                pct = (sig['swing_low'] - price) / sig['swing_low'] * 100
                print(f"[SMC][{inst_id}] СЛИШКОМ ГЛУБОКИЙ ПРОБОЙ! {price:.4f} < {sig['swing_low']*0.95:.4f} ({pct:.1f}%). Сигнал инвалидирован.")
                if self.gui_queue:
                    sym_str = str(inst_id).split('.')[0]
                    self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} пробой swing_low -{pct:.1f}% > 5%, зона сломана", "color": "c-r"}))
                data['dead_signals'].add(round(sig['swing_low'], 8))
                data['pending_signal'] = None
            # Инвалидация 4: Цена ниже OTE зоны (pump-dump / полный разворот)
            elif price < sig['ote_low']:
                print(f"[SMC][{inst_id}] ЦЕНА НИЖЕ OTE ЗОНЫ! {price:.4f} < ote_low {sig['ote_low']:.4f}. Зона пройдена.")
                if self.gui_queue:
                    sym_str = str(inst_id).split('.')[0]
                    self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} цена ниже OTE зоны — полный разворот", "color": "c-r"}))
                data['dead_signals'].add(round(sig['swing_low'], 8))
                data['pending_signal'] = None
            elif not data['in_position'] and len(data['pending_entries']) == 0:
                
                # ВЫСТАВЛЕНИЕ УПРЕЖДАЮЩЕЙ ЛИМИТКИ (ADVANCE PLACEMENT)
                # Вычисляем модель заранее и ставим LIMIT
                proba = self._model_proba_partial(inst_id, sig, partial)
                data['last_model_proba'] = proba
                if proba >= self.model_threshold:
                    sl = sig['stop_loss']
                    tp = sig['tp_aggressive'] if proba >= 0.75 else sig['tp_standard']
                    success = self._nautilus_enter_limit_advance(inst_id, sig['entry_price'], sl, tp, sig)
                    if success:
                        data['pending_signal'] = None
                else:
                    pass # Ждем, пока модель не даст добро. НЕ удаляем сигнал!
            else:
                data['pending_signal'] = None # Сброс, так как лимитка уже висит или в позиции

        # Б) Динамическая Защита выставленных лимиток
        if data['pending_entries'] and partial:
            to_cancel = []
            for client_id, (sig, sl, tp) in data['pending_entries'].items():
                entry_price = sig['entry_price']

                # Инвалидация 1: свеча закрылась ниже swing_low (слом структуры)
                if last_closed_price < sig['swing_low']:
                    data['dead_signals'].add(round(sig['swing_low'], 8))
                    print(f"[SMC][{inst_id}] ЛИМИТ ОТМЕНЕН: СЛОМ СТРУКТУРЫ (закрытие < {sig['swing_low']:.4f})")
                    to_cancel.append(client_id)
                    if self.gui_queue:
                        sym_str = str(inst_id).split(".")[0]
                        self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} Лимит отменен: слом структуры (закрытие ниже зоны).", "color": "c-r"}))
                        self.gui_queue.put(("signal", {
                            "symbol": sym_str, "action": "exit", "reason": "canc",
                            "pnl": 0.0, "total_pnl": data.get('paper_pnl', 0.0)
                        }))
                    continue

                # Инвалидация 2: экстремальный пробой внутри тика (>5%)
                if price < sig['swing_low'] * 0.95:
                    data['dead_signals'].add(round(sig['swing_low'], 8))
                    print(f"[SMC][{inst_id}] ЛИМИТ ОТМЕНЕН: ГЛУБОКИЙ ПРОБОЙ (>5%)")
                    to_cancel.append(client_id)
                    if self.gui_queue:
                        sym_str = str(inst_id).split(".")[0]
                        self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} Лимит отменен: глубокий пробой ниже зоны (>5%).", "color": "c-r"}))
                        self.gui_queue.put(("signal", {
                            "symbol": sym_str, "action": "exit", "reason": "canc",
                            "pnl": 0.0, "total_pnl": data.get('paper_pnl', 0.0)
                        }))
                    continue

                # Инвалидация 3: перехай или касание TP — движение ушло без нас (не отработали)
                if price > sig['swing_high'] or price >= tp:
                    data['dead_signals'].add(round(sig['swing_low'], 8))
                    print(f"[SMC][{inst_id}] ЛИМИТ ОТМЕНЕН: ЦЕНА УШЛА БЕЗ НАС (TP/High достигнут {price:.4f})")
                    to_cancel.append(client_id)
                    
                    if self.gui_queue:
                        sym_str = str(inst_id).split(".")[0]
                        self.gui_queue.put(("log_msg", {"msg": f"[SMC] {sym_str} Лимит отменен: Цена ушла до TP без нас.", "color": "c-r"}))
                        
                        # Косметическое удаление лимитки из UI (имитируем exit)
                        self.gui_queue.put(("signal", {
                            "symbol": sym_str,
                            "action": "exit",
                            "reason": "canc",
                            "pnl": 0.0,
                            "total_pnl": data.get('paper_pnl', 0.0)
                        }))
                    continue
                
                # Проверяем расстояние от текущей цены до нашей лимитки
                distance_pct = abs(price - entry_price) / entry_price * 100.0
                
                # Если цена подошла ближе чем на 1% к лимитке, начинаем активно пинговать модель
                if distance_pct <= 1.0:
                    proba = self._model_proba_partial(inst_id, sig, partial)
                    data['last_model_proba'] = proba
                    
                    # Логирование в UI
                    import time
                    now_sec = time.time()
                    if now_sec - data.get('last_log_time', 0) > 60:
                        data['last_log_time'] = now_sec
                        if self.gui_queue:
                            sym_str = str(inst_id).split(".")[0]
                            if proba >= self.model_threshold:
                                msg = f"[SMC] {sym_str} ПОДХОД К ЛИМИТУ. Модель ЗА ({proba*100:.0f}% \u2265 {self.model_threshold*100:.0f}%)"
                                self.gui_queue.put(("log_msg", {"msg": msg, "color": "c-g"}))
                            else:
                                msg = f"[SMC] {sym_str} ОПАСНОСТЬ! Модель против ({proba*100:.0f}% < {self.model_threshold*100:.0f}%)"
                                self.gui_queue.put(("log_msg", {"msg": msg, "color": "c-y"}))
                                
                    # Если модель видит слом структуры/аномалию на подходе - ОТМЕНЯЕМ ЛИМИТКУ!
                    if proba < self.model_threshold:
                        print(f"[SMC][{inst_id}] МОДЕЛЬ ОТМЕНИЛА ВХОД! probability={proba:.2f} < {self.model_threshold}")
                        to_cancel.append(client_id)
            
            for cid in to_cancel:
                order = self.cache.order(cid)
                if order:
                    self.cancel_order(order)
                    
                # Убираем из ожидания и записываем в лог отмену
                sig, sl, tp = data['pending_entries'].pop(cid, (None, None, None))
                if sig:
                    try:
                        with open("trades_history.csv", "a", encoding="utf-8") as f:
                            dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            f.write(f"{dt_str},{inst_id},CANCELED_BY_MODEL,0.0,0.0,0.0,0.0\n")
                    except: pass

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
            
            # Если сигнал найден, но этот swing_low уже мертв (инвалидирован ранее) — игнорим его
            if signal:
                sl_round = round(signal['swing_low'], 8)
                if sl_round in data['dead_signals']:
                    signal = None
                else:
                    data['pending_signal'] = signal

        if self.gui_queue:
            sym_str = str(inst_id).split(".")[0]
            # Логируем только реальные живые пакеты (после warmup)
            # Убрано сообщение о закрытом пакете для минимизации спама
                
            self._send_metrics(inst_id, packet, signal)
            self.gui_queue.put(("candle", {
                "symbol": sym_str,
                "time":  nanos_to_secs(packet.timestamp),
                "open":  packet.open,  "high": packet.high,
                "low":   packet.low,   "close": packet.close,
            }))

    def _check_signal_invalid(self, signal: dict, current_price: float, last_closed_price: float) -> str | None:
        """
        Проверяет валидность сигнала.
        
        Правила SMC:
        1. Свеча закрылась НА/НИЖЕ swing_low = СЛОМ СТРУКТУРЫ (BOS). ИНВАЛИДЕН.
        2. Живая цена прострелила swing_low до 5% = liquidity grab. ВАЛИДЕН.
        3. Живая цена прострелила swing_low > 5% = слишком глубокий пробой. ИНВАЛИДЕН.
        4. Живая цена ушла ВЫШЕ swing_high = движение пропущено без нас. ИНВАЛИДЕН.
        5. Цена ниже OTE зоны = полный разворот (pump-dump). ИНВАЛИДЕН.
        """
        swing_low  = signal.get('swing_low',  0)
        swing_high = signal.get('swing_high', float('inf'))
        ote_low    = signal.get('ote_low', 0)

        if current_price <= 0 or swing_low <= 0:
            return None

        # Условие 1: Свеча закрылась на/ниже swing_low = BOS (слом структуры)
        if last_closed_price > 0 and last_closed_price <= swing_low:
             return f"свеча закрылась на/ниже swing_low {swing_low:.4f} (настоящий слом структуры)"

        # Условие 2: Экстремальный прострел (больше 5%)
        if current_price < swing_low * 0.95:
            pct_below = (swing_low - current_price) / swing_low * 100
            return f"цена {current_price:.4f} ниже swing_low {swing_low:.4f} на {pct_below:.1f}% (слишком глубокий пробой >5%)"

        # Условие 3: Цена уже ушла ВЫШЕ swing_high = движение пропущено
        if swing_high < float('inf') and current_price > swing_high:
            pct_above = (current_price - swing_high) / swing_high * 100
            return f"цена {current_price:.4f} выше swing_high {swing_high:.4f} на {pct_above:.1f}% — движение пропущено"

        # Условие 4: Цена ниже OTE зоны = pump-dump / полный разворот
        if ote_low > 0 and current_price < ote_low:
            return f"цена {current_price:.4f} ниже OTE зоны {ote_low:.4f} — полный разворот (pump-dump)"

        return None  # зона валидна

    def _register_instrument_from_exchange_info(self, inst_id, exchange_info: dict):
        """Create CurrencyPair from Binance exchangeInfo and register in cache + sandbox."""
        try:
            sym_name = exchange_info.get('symbol', str(inst_id).split('.')[0])
            base_asset = exchange_info.get('baseAsset', '')
            if not base_asset:
                print(f"[HOT_INSERT] exchangeInfo missing baseAsset for {inst_id}")
                return None

            tick_s = "0.00000001"
            step_s = "0.00000001"
            for f in exchange_info.get('filters', []):
                if f['filterType'] == 'PRICE_FILTER':
                    tick_s = f['tickSize']
                if f['filterType'] == 'LOT_SIZE':
                    step_s = f['stepSize']

            tick_s = tick_s.rstrip('0') or '0'
            step_s = step_s.rstrip('0') or '0'
            if tick_s.endswith('.'):
                tick_s += '0'
            if step_s.endswith('.'):
                step_s += '0'

            p_inc = Price.from_str(tick_s)
            s_inc = Quantity.from_str(step_s)
            base_cur = Currency.from_str(base_asset)

            instrument = CurrencyPair(
                instrument_id=inst_id,
                raw_symbol=Symbol(sym_name),
                base_currency=base_cur,
                quote_currency=USDT,
                price_precision=p_inc.precision,
                size_precision=s_inc.precision,
                price_increment=p_inc,
                size_increment=s_inc,
                lot_size=s_inc,
                max_quantity=Quantity(18_000_000_000.0, precision=s_inc.precision),
                min_quantity=Quantity.from_str(step_s),
                max_notional=None,
                min_notional=None,
                max_price=Price(4_000_000_000.0, precision=p_inc.precision),
                min_price=Price.from_str(tick_s),
                margin_init=Decimal("0"),
                margin_maint=Decimal("0"),
                maker_fee=Decimal("0"),
                taker_fee=Decimal("0"),
                ts_event=self.clock.timestamp_ns(),
                ts_init=self.clock.timestamp_ns(),
            )

            self.cache.add_instrument(instrument)
            print(f"[HOT_INSERT] Registered {sym_name} in cache (base={base_asset}, price_prec={p_inc.precision}, size_prec={s_inc.precision})")

            # НЕ вызываем sandbox.exchange.add_instrument() в рантайме!
            # Это крашит Rust matching engine при параллельной обработке тиков.
            # Все инструменты предрегистрируются ДО node.run_async() в flask_server.py.

            return instrument

        except Exception as e:
            print(f"[HOT_INSERT] Failed to register {inst_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _nautilus_enter_limit_advance(self, inst_id, entry_price_val: float, sl: float, tp: float, sig: dict):

        try:
            data = self.instruments_data[inst_id]
            instrument = self.cache.instrument(inst_id)
            if not instrument: 
                return False

            # Вычисляем размер поинта в монетах для сделки на ~100$
            qty_coins = max(100.0 / entry_price_val, 0.01) if entry_price_val > 0 else 0.01
            qty = instrument.make_qty(qty_coins)
            entry_price_obj = instrument.make_price(entry_price_val)
                
            order = self.order_factory.limit(
                instrument_id=inst_id,
                order_side=OrderSide.BUY,
                price=entry_price_obj,
                quantity=qty,
            )
            data['pending_entries'][order.client_order_id] = (sig, sl, tp)
            self.submit_order(order)
            
            risk_pct = abs(entry_price_val - sl) / entry_price_val * 100
            reward_pct = abs(tp - entry_price_val) / entry_price_val * 100
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            
            pr_str = lambda p: f"{p:.6f}".rstrip('0').rstrip('.') if p < 1 else f"{p:.4f}"
            
            msg = f"[LIMIT] {str(inst_id).split('.')[0]} BUY {pr_str(entry_price_val)} | SL {pr_str(sl)} (-{risk_pct:.1f}%) | TP {pr_str(tp)} (+{reward_pct:.1f}%) | RR 1:{rr_ratio:.1f}"
            
            print(f"[NAUTILUS][{inst_id}] ADVANCE LIMIT PLACED -> {msg}")
            if self.gui_queue:
                self.gui_queue.put(("log_msg", {"msg": msg, "color": "c-y"}))
                self.gui_queue.put(("signal", {
                    "symbol": str(inst_id).split(".")[0],
                    "action": "limit",
                    "price": entry_price_val,
                    "sl": sl,
                    "tp": tp,
                    "risk_pct": risk_pct,
                    "reward_pct": reward_pct,
                    "rr_ratio": round(rr_ratio, 1)
                }))
            return True
        except Exception as e:
            print(f"[SMC_CRITICAL] Failed to execute advance limit for {inst_id}: {e}")
            if self.gui_queue:
                sym_str = str(inst_id).split(".")[0]
                self.gui_queue.put(("log_msg", {"msg": f"[SMC] ОШИБКА выставления лимит для {sym_str}: {e}", "color": "c-r"}))
            return False

    def on_order_filled(self, event: OrderFilled):
        client_id = event.client_order_id
        inst_id = event.instrument_id
        data = self.instruments_data.get(inst_id)
        if not data: return
        
        if client_id in data['pending_entries']:
            data['in_position'] = True
            sig, sl_price, tp_price = data['pending_entries'].pop(client_id)
            
            # Регистрируем выходы
            qty = event.last_qty
            data['entry_price_val'] = float(event.last_px)
            instrument = self.cache.instrument(inst_id)

            sl_price_obj = instrument.make_price(sl_price)
            tp_price_obj = instrument.make_price(tp_price)

            # Защита от отклонения SL: если bid уже ниже SL (gap-down), корректируем
            try:
                ob = self.cache.order_book(inst_id)
                current_bid = ob.best_bid_price() if ob else None
                if current_bid and float(current_bid) < sl_price:
                    safe_sl = float(current_bid) * 0.999  # чуть ниже bid
                    sl_price_obj = instrument.make_price(safe_sl)
                    print(f"[NAUTILUS][{inst_id}] Аварийный SL: bid={float(current_bid):.4f} < SL={sl_price:.4f}, сдвигаем SL -> {safe_sl:.4f}")
            except Exception:
                pass

            sl_order = self.order_factory.stop_market(inst_id, OrderSide.SELL, qty, trigger_price=sl_price_obj)
            tp_order = self.order_factory.limit(inst_id, OrderSide.SELL, qty, price=tp_price_obj)

            self.submit_order(sl_order)
            self.submit_order(tp_order)
            data['active_sl_id'] = sl_order.client_order_id
            data['active_tp_id'] = tp_order.client_order_id

            print(f"[NAUTILUS][{inst_id}] ENTRY FILLED @ {data['entry_price_val']:.4f}. Placed SL={sl_price:.4f}, TP={tp_price:.4f}")

            # --- Логируем баланс после входа ---
            try:
                acc = self.cache.account_for_venue(Venue("BINANCE"))
                if acc:
                    usdt_bal = acc.balance(Currency.from_str("USDT"))
                    print(f"[TEST] БАЛАНС ПОСЛЕ ВХОДА: {usdt_bal.free.as_double():.2f} USDT")
            except Exception as _e:
                print(f"[TEST] Баланс недоступен: {_e}")
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
            # BUG 13 FIX: не хардкодим BTC, используем USDT (общий для всех пар)
            try:
                acc = self.cache.account_for_venue(Venue("BINANCE"))
                if acc:
                    usdt_bal = acc.balance(Currency.from_str("USDT"))
                    print(f"[TEST] БАЛАНС ПОСЛЕ ВЫХОДА: {usdt_bal.free.as_double():.2f} USDT")
            except Exception as _e:
                print(f"[TEST] Баланс недоступен: {_e}")
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
            ote_width = signal['ote_high'] - signal['ote_low']
            ote = (cur - signal['ote_low']) / ote_width if ote_width > 0 else 0.0
            
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
        inst_id = event.instrument_id
        if inst_id in self.instruments_data:
            data = self.instruments_data[inst_id]
            # Если отменился ожидающий вход (например, другой системой или нами)
            if event.client_order_id in data['pending_entries']:
                data['pending_entries'].pop(event.client_order_id, None)

    def on_order_rejected(self, event):
        print(f"[NAUTILUS][{event.instrument_id}] ORDER REJECTED: {event.client_order_id} - {event.reason}")
        inst_id = event.instrument_id
        if inst_id in self.instruments_data:
            data = self.instruments_data[inst_id]
            if event.client_order_id in data['pending_entries']:
                data['pending_entries'].pop(event.client_order_id, None)

    def on_order_accepted(self, event):
        pass # Убираем спам в консоль

    def _send_metrics(self, inst_id, packet: Packet, signal: dict | None):
        try:
            data = self.instruments_data[inst_id]
            er = calculate_er(list(data['close_prices']), period=20)

            current_signal = data['pending_signal'] or signal
            if not current_signal and data['pending_entries']:
                first_cid = next(iter(data['pending_entries']))
                current_signal, _, _ = data['pending_entries'][first_cid]
                
            ote_dist, retr = None, None
            price = packet.close

            if current_signal:
                ote_high = current_signal['ote_high']
                ote_low = current_signal['ote_low']
                # Правильный расчёт расстояния до OTE зоны
                if price > ote_high:
                    ote_dist = (price - ote_high) / ote_high * 100
                elif price < ote_low:
                    ote_dist = (ote_low - price) / ote_low * -100
                else:
                    ote_dist = 0.0
                retr = (current_signal['bos_high'] - price) / max(current_signal['bos_high'], 1e-9) * 100

            # Если пакетов много (после загрузки истории), warmup_pct должен быть 100
            total_pkts = len(data['packets'])
            wp = round(total_pkts / data['warmup'] * 100, 1) if total_pkts < data['warmup'] else 100.0

            res = {
                "symbol": str(inst_id).split(".")[0],
                "imbalance": round(packet.imbalance * 100, 2),
                "vpin": round(packet.vpin * 100, 2),
                "er": round(er * 100, 1),
                "volatility": round(packet.candle_range * 100, 4),
                "bucket": int(data['packet_builder'].bucket_usd) if data['packet_builder'] else 0,
                "ote_distance": round(ote_dist, 2) if ote_dist is not None else None,
                "retracement_pct": round(retr, 2) if retr is not None else None,
                "warmup_pct": wp,
                "packets_count": total_pkts,
                "paper_pnl": round(data['paper_pnl'], 2),
                "in_position": data['in_position'],
                "has_signal": current_signal is not None,
                "model_proba": data.get('last_model_proba')
            }
            if current_signal:
                # ... (rest of signal fields)
                res.update({
                    "entry_705": round(current_signal['entry_price'], 6),
                    "ote_high": round(current_signal['ote_high'], 6),
                    "ote_low": round(current_signal['ote_low'], 6),
                    "structural_sl": round(current_signal['stop_loss'], 6),
                    "structural_tp": round(current_signal['tp_standard'], 6),
                    "tp_conservative": round(current_signal['tp_conservative'], 6),
                    "tp_aggressive": round(current_signal['tp_aggressive'], 6),
                    "swing_high": round(current_signal['swing_high'], 6),
                    "swing_low": round(current_signal['swing_low'], 6),
                    "range_size": round(current_signal['range_size'], 6),
                })
                risk = abs(current_signal['entry_price'] - current_signal['stop_loss'])
                reward = abs(current_signal['tp_standard'] - current_signal['entry_price'])
                res["rr_ratio"] = round(reward / risk, 1) if risk > 0 else 0
            else:
                for k in ["entry_705","ote_high","ote_low","structural_sl","structural_tp","tp_conservative","tp_aggressive","swing_high","swing_low","range_size","rr_ratio"]:
                    res[k] = None

            if self.gui_queue:
                self.gui_queue.put(("metrics", res))
        except Exception as e:
            print(f"[SMC] Error in _send_metrics for {inst_id}: {e}")

    def _send_partial(self, inst_id, partial: PartialPacket):
        # BUG 10 FIX: Throttle — макс 2 обновления/сек на символ (иначе Socket.IO захлебывается при 20+ монетах)
        import time as _time
        data = self.instruments_data[inst_id]
        now = _time.time()
        if now - data.get('_last_partial_ts', 0) < 0.5:
            return
        data['_last_partial_ts'] = now
        
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
