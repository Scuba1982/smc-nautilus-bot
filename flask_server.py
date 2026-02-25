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

import io
if sys.stdout:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import yaml
import requests as _requests
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
}
_STATE_LOCK = threading.Lock()


# ─── Маршруты ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/scan")
def api_scan():
    """Возвращает топ-30 монет по объему и волатильности напрямую с бэкенда"""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
        if r.status_code != 200: return jsonify([]), 500
        data = r.json()
        
        # Фильтруем USDT пары, исключаем стейблкоины и BTC
        stablecoins = ("FDUSDUSDT", "USDCUSDT", "TUSDUSDT", "EURUSDT", "BUSDUSDT", "DAIUSDT", "USDPUSDT")
        # ТЕСТ: Строго берем 2 монеты для проверки логов (по просьбе пользователя)
        list_items = [t for t in data if t['symbol'] in ("LAUSDT", "DENTUSDT")]

        # Сортировка по волатильности (range)
        list_items.sort(key=lambda t: (float(t['highPrice']) - float(t['lowPrice'])) / (float(t['openPrice']) or 1), reverse=True)

        # Возвращаем Топ-40 монет для фронтенда и Оркестратора
        top40 = list_items[:40]
        print(f"[FLASK] Backend Scan: Found {len(list_items)} pairs (>10M vol), returning top {len(top40)}.")
        return jsonify(top40)
    except Exception as e:
        print(f"[FLASK] /api/scan error: {e}")
        return jsonify({"error": str(e)}), 500
# ─── Queue → Socket.IO ────────────────────────────────────────────────────────
def queue_reader(gui_queue: queue.Queue):
    """Читает события из gui_queue и эмитирует в браузер."""
    while True:
        try:
            item = gui_queue.get(timeout=1.0)
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            event_type, payload = item[0], item[1]
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
                    if not self.exchange.cache.instrument(inst_id):
                        if inst_id not in self._warned:
                            print(f"[HOT_SANDBOX] Игнорируем тики {inst_id} т.к. инструмента нет в кэше!")
                            self._warned.add(inst_id)
                        # Игнорируем быстрые тики, пока инструмент скачивается
                        return
                super().on_data(data)

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
            
            # Пауза для соблюдения лимитов Binance
            await asyncio.sleep(0.5)

        node_config = TradingNodeConfig(
            trader_id="SEISMIC-SMC-MULTI",
            logging=LoggingConfig(log_level="INFO"),
            exec_engine=LiveExecEngineConfig(reconciliation=False),
            data_clients={BINANCE: BinanceDataClientConfig(
                api_key=api_key, api_secret=api_secret,
                account_type=account_type, testnet=False,
                instrument_provider=InstrumentProviderConfig(
                    load_all=False, load_ids=tuple(load_ids)
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
        )

        node.trader.add_strategy(strategy)
        node.build()
        self.node = node  # Сохраняем ссылку на ноду
        
        # [КРИТИЧЕСКИ ВАЖНО] Инжектим кастомные инструменты в кэш ДО старта,
        # чтобы обойти баги загрузки Binance REST (альткоины могут не находиться или грузиться долго).
        from nautilus_trader.model.instruments import CurrencyPair
        from nautilus_trader.model.currencies import USDT, BTC
        from nautilus_trader.model.identifiers import InstrumentId, Symbol
        from nautilus_trader.model.objects import Price, Quantity
        from decimal import Decimal
        
        for inst_id_str in load_ids:
            inst_id = InstrumentId.from_str(inst_id_str)
            raw_sym = inst_id.symbol.value
            
            instrument = CurrencyPair(
                instrument_id=inst_id,
                raw_symbol=Symbol(raw_sym),
                base_currency=BTC,  # Фейковая, песочнице все равно
                quote_currency=USDT,
                price_precision=8,
                size_precision=8,
                price_increment=Price.from_str("0.00000001"),
                size_increment=Quantity.from_str("0.00000001"), # Максимальная гибкость для песочницы
                lot_size=Quantity.from_str("0.00000001"),
                max_quantity=Quantity.from_str("1000000000.0"),
                min_quantity=Quantity.from_str("0.00000001"),
                max_notional=None,
                min_notional=None,
                max_price=Price.from_str("10000000.0"),
                min_price=Price.from_str("0.00000001"),
                margin_init=Decimal("1.0"),
                margin_maint=Decimal("1.0"),
                maker_fee=Decimal("0.0"),
                taker_fee=Decimal("0.0"),
                ts_event=node.kernel.clock.timestamp_ns(),
                ts_init=node.kernel.clock.timestamp_ns()
            )
            node.kernel.cache.add_instrument(instrument)
            print(f"[HOT_SANDBOX] Симуляция инструмента {inst_id} успешно ИНЖЕКТИРОВАНА в кэш!")

        print(f"[WORKER] Multi-Strategy running for {len(self.symbols)} assets.")
        socketio.emit("strategy_status", {"status": "running", "symbols": self.symbols})
        
        await node.run_async()


# ─── Socket.IO события ────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print("[FLASK] Browser client connected")
    with _STATE_LOCK:
        emit("strategy_status", {
            "status": "running" if _STATE["running"] else "idle",
            "symbols": _STATE.get("symbols", []),
        })


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
    """
    symbols_input = data.get("symbols") or []
    if not symbols_input and data.get("symbol"):
        symbols_input = [data.get("symbol")]

    symbols = [s.strip().upper() for s in symbols_input if s.strip()]
    if not symbols:
        emit("strategy_error", {"error": "No symbols provided"})
        return

    with _STATE_LOCK:
        if _STATE["running"]:
            emit("strategy_error", {"error": "Strategy already running"})
            return
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
    print(f"[FLASK] Seismic SMC Terminal → http://localhost:{PORT}")
    print(f"[FLASK] SCAN → select pair → START")
    socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
