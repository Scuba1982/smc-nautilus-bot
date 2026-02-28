"""
flask_server.py — Точка входа для браузерного режима.
Запуск: python flask_server.py

Архитектура (v2 — fix pyo3/tokio crash):
  - Flask/SocketIO работает в фоновом daemon-потоке
  - Nautilus TradingNode запускается в ГЛАВНОМ потоке через asyncio.run()
  - Это КРИТИЧНО: tokio (Rust async runtime) создаёт потоки, в которых
    Python GIL не инициализирован. Если Nautilus НЕ в главном потоке —
    pyo3 panic: "The Python interpreter is not initialized".

Флоу:
  1. Сервер стартует — Flask в daemon-потоке, главный поток ждёт команды
  2. Браузер: SCAN → выбрать пару → START
  3. socket.emit("start_strategy") → Flask кладёт команду в _NAUTILUS_CMD_QUEUE
  4. Главный поток подхватывает → считает bucket → запускает Nautilus
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
    "gui_queue": None,
    "manual_order_queue": None,  # ручные ордера UI → стратегия
    "snapshot": {},              # symbol -> {metrics, positions, signals} для реконнекта
    "symbols": [],               # список активных символов
}
_STATE_LOCK = threading.Lock()

# Очередь команд Flask → Main thread (для запуска Nautilus из главного потока)
_NAUTILUS_CMD_QUEUE = queue.Queue()

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
        # Фильтруем: только USDT-пары, объём > 10M, не в исключениях и не в бан-листе Nautilus
        list_items = [
            t for t in data
            if t['symbol'].endswith('USDT')
            and t['symbol'] not in EXCLUDED
            and t['symbol'] not in NAUTILUS_SKIP
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


# ─── Nautilus Async (MAIN THREAD) ─────────────────────────────────────────────
async def _run_nautilus_async(config: dict, symbols: tuple, gui_queue: queue.Queue,
                              manual_order_queue: queue.Queue):
    """
    Запускает TradingNode в главном потоке через asyncio.
    КРИТИЧНО: эта функция ДОЛЖНА выполняться в главном потоке Python,
    иначе tokio worker threads вызовут pyo3 panic.
    """
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

    cfg = config
    account_type = BinanceAccountType.SPOT

    symbols_configs = {}

    print(f"[NAUTILUS] Initializing Multi-Instrument for {symbols} (main thread)")

    for i, symbol in enumerate(symbols):
        inst_id = f"{symbol}.BINANCE"

        try:
            socketio.emit("strategy_status", {"status": "loading", "symbol": symbol, "index": i, "total": len(symbols)})
            print(f"[NAUTILUS] Calculating bucket for {symbol} ({i+1}/{len(symbols)})...")

            # Адаптивный бакет — блокирующий вызов, оборачиваем в to_thread
            _inst_id = inst_id  # захват для lambda
            bucket_usd = await asyncio.to_thread(
                lambda: get_adaptive_bucket(_inst_id, testnet=False)
            )

            # Загрузка истории ПОСЛЕДОВАТЕЛЬНО
            def progress_cb(pct, msg, sym=symbol):
                if gui_queue:
                    gui_queue.put(("hist_progress", {"symbol": sym, "pct": pct, "msg": msg}))

            print(f"[NAUTILUS] Загрузка истории для {inst_id}...")
            hist_packets, pb = await asyncio.to_thread(
                lambda: fetch_historical_packets(
                    symbol=inst_id,
                    bucket_usd=bucket_usd,
                    days=1.0,
                    testnet=False,
                    on_progress=progress_cb
                )
            )

            symbols_configs[inst_id] = {
                'bucket_usd': bucket_usd,
                'hist_packets': hist_packets,
                'packet_builder': pb
            }
        except Exception as e:
            # CRASH PROTECTION: Один упавший инструмент НЕ убивает весь батч
            print(f"[NAUTILUS][CRITICAL] ОШИБКА загрузки {symbol}: {type(e).__name__}: {e}")
            traceback.print_exc()
            if gui_queue:
                gui_queue.put(("log_msg", {"msg": f"[ERROR] {symbol} пропущен: {e}", "color": "c-r"}))
            continue  # Переходим к следующей монете

        # Агрессивное соблюдение лимитов Binance
        if i > 0 and i % 5 == 0:
            print(f"[NAUTILUS] Достигли монеты {i}. Жесткая пауза 60 секунд (Binance WAF)...")
            if gui_queue:
                gui_queue.put(("hist_progress", {"symbol": symbol, "pct": 100, "msg": f"Охлаждение API (60с)... [{i}/{len(symbols)}]"}))
            await asyncio.sleep(60.0)
        else:
            await asyncio.sleep(3.0)

    # --- NODE CONFIG ---
    # api_key=None / api_secret=None — публичные данные, без аутентификации
    # (per official Nautilus examples: binance_data_tester.py)
    #
    # load_all=True — провайдер сам вызовет GET /exchangeInfo (без параметра symbols)
    # и загрузит ВСЕ инструменты. Это один HTTP запрос, ~2-3 секунды.
    # Раньше это крашило из-за pyo3 panic в secondary thread — теперь мы в main thread.
    # 13 монет из NAUTILUS_SKIP (maxQty > uint64) — провайдер пропустит их с WARNING.
    print("[NAUTILUS] Provider: load_all=True (все инструменты через один exchangeInfo)")

    node_config = TradingNodeConfig(
        trader_id="SEISMIC-SMC-MULTI",
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=False),
        data_clients={BINANCE: BinanceDataClientConfig(
            api_key=None, api_secret=None,
            account_type=account_type, testnet=False,
            instrument_provider=InstrumentProviderConfig(
                load_all=True,
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
        gui_queue=gui_queue,
        manual_order_queue=manual_order_queue,
        sandbox_client_ref=_sandbox_client_ref,
    )

    node.trader.add_strategy(strategy)
    node.build()

    # PRE-REGISTRATION больше НЕ нужна:
    # BinanceSpotInstrumentProvider загрузит ВСЕ ~430 USDT инструментов через load_ids.
    # SandboxExecutionClient.connect() автоматически добавит их из cache в sandbox exchange.
    # Hot-insert просто подписывается на тики — инструмент уже зарегистрирован.

    print(f"[NAUTILUS] Multi-Strategy running for {len(symbols)} assets (main thread).")
    socketio.emit("strategy_status", {"status": "running", "symbols": list(symbols)})

    try:
        await node.run_async()
    except Exception as node_err:
        print(f"[NAUTILUS][CRITICAL] node.run_async() CRASHED!")
        print(f"[NAUTILUS][CRITICAL] Error: {type(node_err).__name__}: {node_err}")
        traceback.print_exc()
        socketio.emit("strategy_status", {"status": "error", "error": f"Node crash: {node_err}"})


async def _nautilus_event_loop():
    """
    Главный event loop в MAIN THREAD.
    Ожидает команды из Flask (через _NAUTILUS_CMD_QUEUE) и запускает Nautilus.
    """
    print("[MAIN] Nautilus event loop started (main thread, asyncio)")

    while True:
        # Non-blocking проверка очереди команд
        try:
            cmd = _NAUTILUS_CMD_QUEUE.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

        action = cmd.get("action")

        if action == "start_nautilus":
            try:
                await _run_nautilus_async(
                    cmd["config"], cmd["symbols"],
                    cmd["gui_queue"], cmd["manual_order_queue"]
                )
            except Exception as e:
                print(f"[NAUTILUS] Error: {type(e).__name__}: {e}")
                traceback.print_exc()
                socketio.emit("strategy_status", {"status": "error", "error": str(e)})
            finally:
                with _STATE_LOCK:
                    _STATE["running"] = False
                print("[MAIN] Nautilus finished. Running=False. Ready for next batch.")
                socketio.emit("strategy_status", {
                    "status": "batch_done",
                    "symbols": list(cmd.get("symbols", []))
                })


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

    # Если Nautilus уже работает — динамически добавляем монеты через hot-insert
    with _STATE_LOCK:
        running = _STATE["running"]

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

                        # exchangeInfo больше НЕ нужен — провайдер уже загрузил ВСЕ инструменты
                        mq = _STATE.get("manual_order_queue")
                        if mq:
                            mq.put({
                                "action": "add_instrument_ready",
                                "symbol": sym,
                                "bucket_usd": bucket_usd,
                                "hist_packets": hist_packets,
                                "packet_builder": pb,
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

    # Отправляем команду в главный поток (Nautilus ДОЛЖЕН быть в main thread)
    _NAUTILUS_CMD_QUEUE.put({
        "action": "start_nautilus",
        "config": CONFIG,
        "symbols": tuple(symbols),
        "gui_queue": gui_queue,
        "manual_order_queue": manual_order_queue,
    })


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[FLASK] Seismic SMC Terminal -> http://localhost:{PORT}")
    print(f"[FLASK] SCAN -> select pair -> START")
    print(f"[FLASK] Architecture: Flask=daemon thread, Nautilus=MAIN thread (asyncio)")

    # Flask/SocketIO в фоновом daemon-потоке (Flask thread-safe с async_mode="threading")
    flask_thread = threading.Thread(
        target=lambda: socketio.run(
            app, host=HOST, port=PORT,
            debug=False, use_reloader=False, allow_unsafe_werkzeug=True
        ),
        daemon=True,
    )
    flask_thread.start()
    print(f"[FLASK] Server running in background thread on http://localhost:{PORT}")

    # Nautilus в ГЛАВНОМ потоке через asyncio.run()
    # КРИТИЧНО: tokio/pyo3 требуют main thread для корректной инициализации Python GIL.
    # Все предыдущие краши (pyo3 panic, crash на 2-3 монете) были из-за
    # запуска Nautilus в threading.Thread.
    try:
        asyncio.run(_nautilus_event_loop())
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C — завершение...")
    except Exception as e:
        print(f"[MAIN] Fatal error: {type(e).__name__}: {e}")
        traceback.print_exc()
