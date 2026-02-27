"""
flask_server.py — Точка входа для браузерного режима.
Запуск: python flask_server.py

Флоу:
  1. Сервер стартует — Nautilus НЕ запускается
  2. Браузер: SCAN → выбрать пару → START
  3. Браузер отправляет socket.emit("start_strategy", {symbol: "SOLUSDT"})
  4. Сервер считает adaptive bucket для монеты → запускает Nautilus
  5. Данные текут в браузер: candle / partial / metrics
"""
import os
import sys

# ОБЯЗАТЕЛЬНО до всех других импортов
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import queue
import threading
import asyncio
import requests
import json
import time
import traceback

# 13 монет с maxQty > 18.4B — Nautilus не может загрузить (uint64 limit в Quantity)
NAUTILUS_SKIP = {
    "1000CHEEMSUSDT", "1000SATSUSDT", "1MBABYDOGEUSDT", "BOMEUSDT", "BONKUSDT",
    "BTTCUSDT", "DOGSUSDT", "FLOKIUSDT", "HMSTRUSDT", "NEIROUSDT",
    "PEPEUSDT", "SHIBUSDT", "XECUSDT",
}

import io
if sys.stdout:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import yaml
from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO, emit

# ─── Flask / SocketIO ─────────────────────────────────────────────────────────
app = Flask(__name__, static_url_path="", static_folder=".")
app.config["SECRET_KEY"] = "seismic-smc-key"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ─── Конфиг ──────────────────────────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

flask_cfg = CONFIG.get("flask", {})
HOST = flask_cfg.get("host", "0.0.0.0")
PORT = int(flask_cfg.get("port", 5000))

# ─── Состояние сервера ────────────────────────────────────────────────────────
_STATE = {
    "running": False,            # запущена ли стратегия
    "symbol": None,              # "SOLUSDT"
    "instrument_id": None,       # "SOLUSDT.BINANCE"
    "bucket_usd": None,
    "worker": None,              # _FlaskNautilusWorker
    "gui_queue": None,
    "manual_order_queue": None,  # ручные ордера UI → стратегия
    "snapshot": {},              # symbol -> {metrics, positions, signals} для реконнекта
    "symbols": [],               # список активных символов
}
_STATE_LOCK = threading.Lock()

# Единая очередь для hot-insert монет (вместо 40 параллельных потоков)
_HOT_LOAD_QUEUE = queue.Queue()
_HOT_LOADER_RUNNING = False


# ─── Маршруты ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/scan")
def api_scan():
    """Возвращает топ-50 монет по объему и волатильности напрямую с бэкенда"""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
        if r.status_code != 200: return jsonify([]), 500
        data = r.json()

        # Исключаем BTC, стейблкоины и другой нежелательный мусор
        EXCLUDED = {
            "BTCUSDT",                                          # BTC: % волатильность низкая — OTE хуже работает
            "FDUSDUSDT", "USDCUSDT", "TUSDUSDT", "EURUSDT",   # стейблкоины
            "BUSDUSDT", "DAIUSDT", "USDPUSDT", "USDSUSDT", "SUSDUSDT", "GBPUSDT",
            "BTCSTUSDT", "SHIBUSDT",                           # дробные/мусорные монеты
        }
        # Фильтруем: только USDT-пары, объём > 10M, не в исключениях
        list_items = [
            t for t in data
            if t['symbol'].endswith('USDT')
            and t['symbol'] not in EXCLUDED
            and float(t.get('quoteVolume', 0)) > 10_000_000
        ]

        # Сортировка по волатильности: (High - Low) / Open = % дневного диапазона.
        # Именно диапазон (размах свинга), а не |change%|, — важен для нахождения
        # зон Dealing Range / OTE правильного масштаба.
        list_items.sort(
            key=lambda t: (float(t['highPrice']) - float(t['lowPrice'])) / (float(t['openPrice']) or 1),
            reverse=True
        )

        # Возвращаем топ-50 монет для Оркестратора на фронте
        top50 = list_items[:50]
        print(f"[FLASK] Scan: {len(list_items)} пар (>10M vol). Топ-{len(top50)}.")
        # Дебаг: показываем топ-10 в консоли
        for i, t in enumerate(top50[:10]):
            op = float(t['openPrice']) or 1
            rng = (float(t['highPrice']) - float(t['lowPrice'])) / op * 100
            print(f"  #{i+1} {t['symbol']}: range={rng:.1f}%  vol=${float(t['quoteVolume'])/1e6:.0f}M")
        return jsonify(top50)
    except Exception as e:
        print(f"[FLASK] /api/scan error: {e}")
        return jsonify({"error": str(e)}), 500
# ─── Queue → Socket.IO ────────────────────────────────────────────────────────
def queue_reader(gui_queue: queue.Queue):
    """Читает события из gui_queue и эмитирует в браузер + поддерживает snapshot."""
    while True:
        try:
            item = gui_queue.get(timeout=1.0)
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            event_type, payload = item[0], item[1]

            # ── Обновляем snapshot для персистентности при реконнекте ──
            if event_type == "metrics" and isinstance(payload, dict):
                sym = payload.get("symbol")
                if sym:
                    with _STATE_LOCK:
                        if sym not in _STATE["snapshot"]:
                            _STATE["snapshot"][sym] = {}
                        _STATE["snapshot"][sym]["metrics"] = payload

            elif event_type == "signal" and isinstance(payload, dict):
                sym = payload.get("symbol")
                if sym:
                    with _STATE_LOCK:
                        if sym not in _STATE["snapshot"]:
                            _STATE["snapshot"][sym] = {}
                        action = payload.get("action")
                        if action in ("limit", "enter"):
                            _STATE["snapshot"][sym]["signal"] = payload
                        elif action == "exit":
                            # Позиция закрыта — убираем сигнал из snapshot
                            _STATE["snapshot"][sym].pop("signal", None)

            if event_type in ("candle", "partial", "metrics", "bucket_info",
                              "multi_bucket_info", "signal", "hist_progress",
                              "candles_batch", "log_msg",
                              "order_response", "cancel_response",
                              "account_update", "commission_update",
                              "order_book"):
                socketio.emit(event_type, payload)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[QUEUE] Error: {e}")


# ─── Nautilus Worker ──────────────────────────────────────────────────────────
class _FlaskNautilusWorker(threading.Thread):
    """Запускает TradingNode в отдельном потоке без Qt."""

    def __init__(self, config: dict, symbols: list, gui_queue: queue.Queue,
                 manual_order_queue: queue.Queue = None):
        super().__init__(daemon=True)
        self.config              = config
        self.symbols             = symbols          # список монет
        self.gui_queue           = gui_queue
        self.manual_order_queue  = manual_order_queue

    def run(self):
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            print(f"[FLASK][WORKER ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit("strategy_status", {"status": "error", "error": str(e)})
        finally:
            # Сбрасываем флаг — позволяем следующему батчу запуститься
            with _STATE_LOCK:
                _STATE["running"] = False
                _STATE["worker"]  = None
            print("[FLASK] Worker finished. Running=False. Ready for next batch.")
            socketio.emit("strategy_status", {"status": "batch_done",
                                              "symbols": self.symbols})

    async def _run_async(self):
        from live_packet_builder import get_adaptive_bucket, fetch_historical_packets
        from nautilus_trader.live.node import TradingNode
        from nautilus_trader.config import (
            TradingNodeConfig, InstrumentProviderConfig, LoggingConfig
        )
        from nautilus_trader.adapters.binance import (
            BINANCE, BinanceAccountType, BinanceDataClientConfig,
            BinanceLiveDataClientFactory,
        )
        from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
        from nautilus_trader.config import LiveExecEngineConfig
        from smc_nautilus_strategy import MultiSMCStrategy

        from nautilus_trader.live.factories import LiveExecClientFactory
        from nautilus_trader.adapters.sandbox.execution import SandboxExecutionClient
        from nautilus_trader.model.identifiers import Venue
        from nautilus_trader.model.data import QuoteTick, TradeTick
        from nautilus_trader.model.objects import Price, Quantity
        from nautilus_trader.model.enums import AggressorSide
        from decimal import Decimal
        from nautilus_trader.backtest.models.fee import MakerTakerFeeModel
        from nautilus_trader.model.currencies import USDT
        class HotSandboxExecutionClient(SandboxExecutionClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._warned = set()

            def on_data(self, data) -> None:
                # Фикс краша: Websocket от Binance запускается БЫСТРЕЕ, 
                # чем REST HTTP успевает положить Instrument в кэш. Убивает Sandbox.
                if hasattr(data, "instrument_id"):
                    inst_id = data.instrument_id
                    if not self._cache.instrument(inst_id):
                        if inst_id not in self._warned:
                            print(f"[HOT_SANDBOX] Игнорируем тики {inst_id} т.к. инструмента нет в кэше!")
                            self._warned.add(inst_id)
                        # Игнорируем быстрые тики, пока инструмент скачивается
                        return
                super().on_data(data)

        # Сохраняем ссылку на sandbox client для регистрации инструментов
        _sandbox_client_ref = [None]
        
        class SandboxLiveExecClientFactory(LiveExecClientFactory):
            @staticmethod
            def create(loop, portfolio, msgbus, cache, clock, config, **kwargs) -> HotSandboxExecutionClient:
                client = HotSandboxExecutionClient(
                    loop=loop,
                    portfolio=portfolio,
                    msgbus=msgbus,
                    cache=cache,
                    clock=clock,
                    config=config
                )
                _sandbox_client_ref[0] = client  # Сохраняем!
                return client

        cfg        = self.config
        testnet    = cfg.get("testnet", True)
        api_key    = cfg.get("api_key")    or os.environ.get("BINANCE_API_KEY")    or None
        api_secret = cfg.get("api_secret") or os.environ.get("BINANCE_API_SECRET") or None
        account_type = BinanceAccountType.SPOT

        symbols_configs = {}
        load_ids = [] # Изначально пустой список, будем добавлять только то, что выбрал юзер
        
        print(f"[WORKER] Initializing Multi-Instrument Nautilus for {self.symbols}")
        
        for i, symbol in enumerate(self.symbols):
            inst_id = f"{symbol}.BINANCE"
            
            try:
                load_ids.append(inst_id)
                
                socketio.emit("strategy_status", {"status": "loading", "symbol": symbol, "index": i, "total": len(self.symbols)})
                print(f"[WORKER] Calculating bucket for {symbol} ({i+1}/{len(self.symbols)})...")
                
                # Адаптивный бакет ( Rate Limited )
                bucket_usd = get_adaptive_bucket(inst_id, testnet=testnet)
                
                # Загрузка истории ПОСЛЕДОВАТЕЛЬНО
                def progress_cb(pct, msg, sym=symbol):
                    if self.gui_queue:
                        self.gui_queue.put(("hist_progress", {"symbol": sym, "pct": pct, "msg": msg}))
                        
                print(f"[WORKER] Загрузка истории для {inst_id}...")
                hist_packets, pb = fetch_historical_packets(
                    symbol=inst_id,
                    bucket_usd=bucket_usd,
                    days=1.0, # 24 часа для полноценного расчета структуры
                    testnet=testnet,
                    on_progress=progress_cb
                )
                
                symbols_configs[inst_id] = {
                    'bucket_usd': bucket_usd,
                    'hist_packets': hist_packets,
                    'packet_builder': pb
                }
            except Exception as e:
                # CRASH PROTECTION: Один упавший инструмент НЕ убивает весь батч
                print(f"[WORKER][CRITICAL] ОШИБКА загрузки {symbol}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                if inst_id in load_ids:
                    load_ids.remove(inst_id)
                if self.gui_queue:
                    self.gui_queue.put(("log_msg", {"msg": f"[ERROR] {symbol} пропущен: {e}", "color": "c-r"}))
                continue  # Переходим к следующей монете
            
            # BUG 1 FIX: Агрессивное соблюдение лимитов Binance
            # Было: пауза каждые 15 монет (60с) + 1с между остальными → краш на 19-21 монете
            # Стало: пауза каждые 5 монет (60с) + 3с между остальными
            if i > 0 and i % 5 == 0:
                print(f"[WORKER] Достигли монеты {i}. Жесткая пауза 60 секунд (отдых от Binance WAF)...")
                if self.gui_queue:
                    self.gui_queue.put(("hist_progress", {"symbol": symbol, "pct": 100, "msg": f"Охлаждение API (60с)... [{i}/{len(self.symbols)}]"}))
                await asyncio.sleep(60.0)
            else:
                await asyncio.sleep(3.0)  # 3 секунды между монетами (было 1с)

        node_config = TradingNodeConfig(
            trader_id="SEISMIC-SMC-MULTI",
            logging=LoggingConfig(log_level="INFO"),
            exec_engine=LiveExecEngineConfig(reconciliation=False),
            data_clients={BINANCE: BinanceDataClientConfig(
                api_key=api_key, api_secret=api_secret,
                account_type=account_type, testnet=False,
                instrument_provider=InstrumentProviderConfig(
                    load_all=False,
                    load_ids=frozenset(load_ids),
                    log_warnings=False,
                ),
            )},
            exec_clients={
                BINANCE: SandboxExecutionClientConfig(
                    venue="BINANCE",
                    account_type="MARGIN",
                    oms_type="NETTING",
                    base_currency="USDT",
                    starting_balances=["1000000 USDT"],
                )
            }
        )

        node = TradingNode(config=node_config)
        node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
        node.add_exec_client_factory(BINANCE, SandboxLiveExecClientFactory)

        strategy = MultiSMCStrategy(
            strategy_config=cfg,
            symbols_configs=symbols_configs,
            gui_queue=self.gui_queue,
            manual_order_queue=self.manual_order_queue,
            sandbox_client_ref=_sandbox_client_ref,
        )

        node.trader.add_strategy(strategy)
        node.build()
        self.node = node  # Сохраняем ссылку на ноду
        
        # --- PRE-REGISTRATION: регистрируем ВСЕ USDT инструменты ДО node.run_async() ---
        # Это критично: sandbox.exchange.add_instrument() НЕЛЬЗЯ вызывать в рантайме
        # (пока exchange обрабатывает тики) — это крашит Rust matching engine.
        # Поэтому регистрируем всё заранее, а hot-insert потом только подписывается на тики.
        try:
            from nautilus_trader.model.instruments import CurrencyPair
            from nautilus_trader.model.identifiers import InstrumentId as _IID, Symbol as _Sym
            from nautilus_trader.model.currencies import USDT as _USDT, Currency as _Cur

            print("[WORKER] Pre-registering ALL USDT instruments from Binance exchangeInfo...")
            ei_resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
            if ei_resp.status_code == 200:
                all_symbols = ei_resp.json().get('symbols', [])
                sandbox = _sandbox_client_ref[0]
                registered = 0
                skipped = 0
                for sym_info in all_symbols:
                    sym_name = sym_info.get('symbol', '')
                    if not sym_name.endswith('USDT') or sym_name in NAUTILUS_SKIP:
                        continue
                    if sym_info.get('status') != 'TRADING':
                        continue

                    inst_id = _IID.from_str(f"{sym_name}.BINANCE")

                    base_asset = sym_info.get('baseAsset', '')
                    if not base_asset:
                        continue

                    tick_s, step_s = "0.00000001", "0.00000001"
                    for f in sym_info.get('filters', []):
                        if f['filterType'] == 'PRICE_FILTER':
                            tick_s = f['tickSize']
                        if f['filterType'] == 'LOT_SIZE':
                            step_s = f['stepSize']

                    tick_s = tick_s.rstrip('0') or '0'
                    step_s = step_s.rstrip('0') or '0'
                    if tick_s.endswith('.'): tick_s += '0'
                    if step_s.endswith('.'): step_s += '0'

                    try:
                        p_inc = Price.from_str(tick_s)
                        s_inc = Quantity.from_str(step_s)
                        base_cur = _Cur.from_str(base_asset)

                        instrument = CurrencyPair(
                            instrument_id=inst_id, raw_symbol=_Sym(sym_name),
                            base_currency=base_cur, quote_currency=_USDT,
                            price_precision=p_inc.precision, size_precision=s_inc.precision,
                            price_increment=p_inc, size_increment=s_inc, lot_size=s_inc,
                            max_quantity=Quantity(18_000_000_000.0, precision=s_inc.precision),
                            min_quantity=Quantity.from_str(step_s),
                            max_notional=None, min_notional=None,
                            max_price=Price(4_000_000_000.0, precision=p_inc.precision),
                            min_price=Price.from_str(tick_s),
                            margin_init=Decimal("0"), margin_maint=Decimal("0"),
                            maker_fee=Decimal("0"), taker_fee=Decimal("0"),
                            ts_event=node.kernel.clock.timestamp_ns(),
                            ts_init=node.kernel.clock.timestamp_ns(),
                        )
                        node.kernel.cache.add_instrument(instrument)
                        if sandbox and hasattr(sandbox, 'exchange'):
                            sandbox.exchange.add_instrument(instrument)
                        registered += 1
                    except Exception:
                        skipped += 1

                print(f"[WORKER] Pre-registered {registered} USDT instruments ({skipped} skipped)")
            else:
                print(f"[WORKER] exchangeInfo HTTP {ei_resp.status_code}, skipping pre-registration")
        except Exception as prereg_err:
            print(f"[WORKER] Pre-registration failed: {prereg_err}")
            import traceback
            traceback.print_exc()
        # --- END PRE-REGISTRATION ---
            
        print(f"[WORKER] Multi-Strategy running for {len(self.symbols)} assets.")
        socketio.emit("strategy_status", {"status": "running", "symbols": self.symbols})
        
        try:
            await node.run_async()
        except Exception as node_err:
            print(f"[WORKER][CRITICAL] node.run_async() CRASHED!")
            print(f"[WORKER][CRITICAL] Error: {type(node_err).__name__}: {node_err}")
            import traceback
            traceback.print_exc()
            socketio.emit("strategy_status", {"status": "error", "error": f"Node crash: {node_err}"})


# ─── Socket.IO события ────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print("[FLASK] Browser client connected")
    with _STATE_LOCK:
        emit("strategy_status", {
            "status": "running" if _STATE["running"] else "idle",
            "symbols": _STATE.get("symbols", []),
        })


@socketio.on("request_state")
def on_request_state():
    """Клиент запрашивает полное состояние после перезагрузки страницы."""
    print("[FLASK] Client requested state snapshot")
    with _STATE_LOCK:
        snapshot = dict(_STATE.get("snapshot", {}))
        running = _STATE.get("running", False)
        symbols = _STATE.get("symbols", [])

    # Отдаём полный снапшот: статус + все метрики + все сигналы
    emit("state_snapshot", {
        "running": running,
        "symbols": symbols,
        "coins": snapshot,   # {symbol: {metrics: {...}, signal: {...}}}
    })
    print(f"[FLASK] Snapshot sent: {len(snapshot)} coins")


@socketio.on("disconnect")
def on_disconnect():
    print("[FLASK] Browser client disconnected")


@socketio.on("set_trade_mode")
def on_set_trade_mode(data):
    """Переключение Paper / Real."""
    mode = data.get("mode", "paper")
    with _STATE_LOCK:
        _STATE["trade_mode"] = mode
    print(f"[FLASK] Trade mode set to: {mode}")
    emit("order_response", {"status": "ok", "msg": f"Режим: {mode.upper()}"})


@socketio.on("submit_order")
def on_submit_order(data):
    """Приём ордера из UI → передаём в стратегию через manual_order_queue."""
    mode   = data.get("mode", "paper")
    side   = data.get("side", "BUY")
    price  = data.get("price", 0)
    qty    = data.get("qty", 0)
    symbol = data.get("symbol", "")
    print(f"[FLASK] ORDER [{mode.upper()}] {side} {qty} @ {price} | {symbol}")

    with _STATE_LOCK:
        mq = _STATE.get("manual_order_queue")

    if mq is not None and _STATE.get("running"):
        mq.put(data)
        # Ответ придёт из стратегии через gui_queue → socketio.emit("order_response", ...)
    else:
        # Стратегия не запущена — бумажная симуляция на стороне Flask
        if mode == "paper":
            emit("order_response", {
                "status": "ok",
                "msg": f"[PAPER offline] {side} {qty} {symbol} @ {price} — стратегия не запущена"
            })
        else:
            emit("order_response", {
                "status": "error",
                "msg": "Real-режим недоступен: сначала запустите стратегию (SCAN → START)"
            })


@socketio.on("update_paper_sltp")
def on_update_paper_sltp(data):
    """Обновление SL/TP открытой paper-позиции из UI (drag или ввод вручную)."""
    sl = data.get("sl", 0)
    tp = data.get("tp", 0)
    print(f"[FLASK] UPDATE_SLTP SL={sl} TP={tp}")
    with _STATE_LOCK:
        mq      = _STATE.get("manual_order_queue")
        running = _STATE.get("running", False)
    if mq is not None and running:
        mq.put({"action": "update_sltp", "sl": sl, "tp": tp})


@socketio.on("cancel_all_orders")
def on_cancel_all_orders(data):
    """Отмена всех ордеров — передаём команду в PanelStrategy через очередь."""
    mode   = data.get("mode", "paper")
    symbol = data.get("symbol", "")
    print(f"[FLASK] CANCEL ALL [{mode.upper()}] {symbol}")

    with _STATE_LOCK:
        mq      = _STATE.get("manual_order_queue")
        running = _STATE.get("running", False)

    if mq is not None and running:
        mq.put({"action": "cancel_all"})
        # Реальный ответ придёт из PanelStrategy через gui_queue → cancel_response
    else:
        emit("cancel_response", {
            "msg": "Стратегия не запущена — нечего отменять"
        })


@socketio.on("start_strategy")
def on_start_strategy(data):
    """
    Браузер прислал: {"symbols": ["SOLUSDT", "BTCUSDT"]}
    Запускаем Nautilus для списка монет.
    Если стратегия уже работает — ждём пока она не освободится.
    """
    symbols_input = data.get("symbols") or []
    if not symbols_input and data.get("symbol"):
        symbols_input = [data.get("symbol")]

    symbols = [s.strip().upper() for s in symbols_input if s.strip()]
    if not symbols:
        emit("strategy_error", {"error": "No symbols provided"})
        return

    # Фильтруем монеты которые Nautilus не может загрузить (maxQty > uint64 limit)
    skipped_initial = [s for s in symbols if s in NAUTILUS_SKIP]
    symbols = [s for s in symbols if s not in NAUTILUS_SKIP]
    if skipped_initial:
        print(f"[SKIP] {len(skipped_initial)} монет пропущено (maxQty limit): {', '.join(skipped_initial)}")
    if not symbols:
        emit("strategy_error", {"error": "All symbols skipped (Nautilus maxQty limit)"})
        return

    # Если предыдущий воркер ещё загружает данные — динамически добавляем монеты в него!
    with _STATE_LOCK:
        running = _STATE["running"]
        prev_worker = _STATE.get("worker")

    if running:
        # Фильтруем монеты которые Nautilus не может загрузить (maxQty > uint64)
        global _HOT_LOADER_RUNNING
        skipped = [s for s in symbols if s in NAUTILUS_SKIP]
        good = [s for s in symbols if s not in NAUTILUS_SKIP]
        if skipped:
            print(f"[SKIP] {len(skipped)} монет пропущено (maxQty limit): {', '.join(skipped)}")
        for sym in good:
            _HOT_LOAD_QUEUE.put(sym)
        print(f"[FLASK] Добавили {len(good)} монет в очередь загрузки (всего в очереди: ~{_HOT_LOAD_QUEUE.qsize()})")
        
        # Запускаем единственный loader-поток если ещё не запущен
        if not _HOT_LOADER_RUNNING:
            _HOT_LOADER_RUNNING = True
            def _hot_loader_worker():
                global _HOT_LOADER_RUNNING
                import time as _time
                from live_packet_builder import get_adaptive_bucket, fetch_historical_packets
                loaded = 0
                while True:
                    try:
                        sym = _HOT_LOAD_QUEUE.get(timeout=5.0)  # Ждём 5с, потом проверяем
                    except queue.Empty:
                        if _HOT_LOAD_QUEUE.empty():
                            print(f"[HOT-LOADER] Очередь пуста. Загружено {loaded} монет. Поток завершён.")
                            _HOT_LOADER_RUNNING = False
                            return
                        continue
                    
                    try:
                        inst_id_str = f"{sym}.BINANCE"
                        testnet = CONFIG.get("testnet", False)
                        bucket_usd = get_adaptive_bucket(inst_id_str, testnet=testnet)
                        
                        def prog_cb(pct, msg, s=sym):
                            gq = _STATE.get("gui_queue")
                            if gq:
                                gq.put(("hist_progress", {"symbol": s, "pct": pct, "msg": msg}))

                        hist_packets, pb = fetch_historical_packets(
                            symbol=inst_id_str, bucket_usd=bucket_usd,
                            days=1.0, testnet=testnet, on_progress=prog_cb
                        )
                        
                        # Fetch exchangeInfo for manual instrument registration
                        exchange_info = None
                        try:
                            ei_resp = requests.get(
                                "https://api.binance.com/api/v3/exchangeInfo",
                                params={"symbol": sym}, timeout=10
                            )
                            if ei_resp.status_code == 200:
                                ei_symbols = ei_resp.json().get('symbols', [])
                                if ei_symbols:
                                    exchange_info = ei_symbols[0]
                        except Exception as ei_err:
                            print(f"[HOT-LOADER] exchangeInfo for {sym} failed: {ei_err}")

                        mq = _STATE.get("manual_order_queue")
                        if mq:
                            mq.put({
                                "action": "add_instrument_ready",
                                "symbol": sym,
                                "bucket_usd": bucket_usd,
                                "hist_packets": hist_packets,
                                "packet_builder": pb,
                                "exchange_info": exchange_info,
                            })
                        loaded += 1
                        print(f"[HOT-LOADER] [{loaded}] {sym} загружен и отправлен в стратегию")
                        
                    except Exception as e:
                        print(f"[HOT-LOADER] Ошибка {sym}: {e}")
                        import traceback
                        traceback.print_exc()
            
            threading.Thread(target=_hot_loader_worker, daemon=True).start()
        
        with _STATE_LOCK:
            _STATE["symbols"] = list(set(_STATE.get("symbols", [])) | set(symbols))
            all_syms = _STATE["symbols"]
            
        emit("strategy_status", {"status": "running", "symbols": all_syms})
        return

    with _STATE_LOCK:
        _STATE["running"] = True
        _STATE["symbols"] = symbols

    print(f"[FLASK] START_STRATEGY received for {len(symbols)} symbols: {symbols}")
    emit("strategy_status", {"status": "starting", "symbols": symbols})

    gui_queue           = queue.Queue()
    manual_order_queue  = queue.Queue()
    _STATE["gui_queue"]           = gui_queue
    _STATE["manual_order_queue"]  = manual_order_queue

    reader = threading.Thread(target=queue_reader, args=(gui_queue,), daemon=True)
    reader.start()

    worker = _FlaskNautilusWorker(CONFIG, tuple(symbols), gui_queue, manual_order_queue)
    _STATE["worker"] = worker
    worker.start()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[FLASK] Seismic SMC Terminal -> http://localhost:{PORT}")
    print(f"[FLASK] SCAN -> select pair -> START")
    socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
