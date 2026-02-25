import os
import asyncio
import threading
from flask import Flask
from flask_socketio import SocketIO
import traceback

from nautilus_trader.config import TradingNodeConfig, LoggingConfig, InstrumentProviderConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.binance import (
    BINANCE, BinanceDataClientConfig, BinanceLiveDataClientFactory
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

# 1. Минимальный Flask + SocketIO + HTML
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>L2 Order Book Test</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        body { background: #000; color: #ccc; font-family: monospace; padding: 20px; }
        .row { display: flex; width: 300px; justify-content: space-between; }
        .ask { color: #ff073a; }
        .bid { color: #39ff14; }
        .spread { color: #fff; font-weight: bold; margin: 10px 0; border-top: 1px solid #333; border-bottom: 1px solid #333; padding: 5px 0;}
    </style>
</head>
<body>
    <h2>Order Book: <span id="sym">Loading...</span></h2>
    <div id="ob">
        <div>Ожидание данных от Nautilus Trader...</div>
    </div>
    <script>
        const socket = io();
        socket.on('order_book', function(data) {
            document.getElementById('sym').innerText = data.symbol;
            let html = '';
            
            // Asks (сортируем по убыванию цены для правильного отображения стакана)
            data.asks.slice().reverse().forEach(ask => {
                html += `<div class="row ask"><span>${ask.price.toFixed(2)}</span><span>${ask.size.toFixed(4)}</span></div>`;
            });
            
            html += `<div class="row spread"><span>SPREAD</span><span>${data.spread.toFixed(2)}</span></div>`;
            
            // Bids (по убыванию цены)
            data.bids.forEach(bid => {
                html += `<div class="row bid"><span>${bid.price.toFixed(2)}</span><span>${bid.size.toFixed(4)}</span></div>`;
            });
            
            document.getElementById('ob').innerHTML = html;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

# 2. Стратегия для подписки на дельты стакана
class OrderBookTestStrategy(Strategy):
    def __init__(self, instrument_id: str):
        super().__init__()
        self.instrument_id = InstrumentId.from_str(instrument_id)
        self.ticks = 0

    def on_start(self):
        print(f"Подписка на стакан для {self.instrument_id}...")
        # Метод для подписки на дельты L2
        self.subscribe_order_book_deltas(self.instrument_id)

    def on_order_book_deltas(self, deltas):
        try:
            self.ticks += 1
            if self.ticks % 10 != 0:  # Троттлинг
                return

            ob = self.cache.order_book(self.instrument_id)
            if ob is None:
                return

            # Вспомогательная функция для извлечения значения (свойство или метод)
            def get_val(obj, attr):
                val = getattr(obj, attr, None)
                if callable(val):
                    return val()
                return val

            # Получаем уровни (всегда как метод в новых версиях)
            asks_levels = list(ob.asks())[:5]
            bids_levels = list(ob.bids())[:5]

            asks = [{"price": float(get_val(a, 'price')), "size": float(get_val(a, 'size'))} for a in asks_levels]
            bids = [{"price": float(get_val(b, 'price')), "size": float(get_val(b, 'size'))} for b in bids_levels]
            
            spread = 0.0
            if asks and bids:
                spread = asks[0]["price"] - bids[0]["price"]

            socketio.emit('order_book', {
                "symbol": str(self.instrument_id),
                "asks": asks,
                "bids": bids,
                "spread": spread
            })
        except Exception as e:
            print(f"FAILED on version V4: {e}")
            traceback.print_exc()

# 3. Запуск Nautilus Node
def run_nautilus():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    inst_id_str = "BTCUSDT.BINANCE"
    
    config = TradingNodeConfig(
        trader_id="TEST-OB",
        logging=LoggingConfig(log_level="INFO"),
        data_clients={
            BINANCE: BinanceDataClientConfig(
                instrument_provider=InstrumentProviderConfig(
                    load_ids=(inst_id_str,)
                )
            )
        }
    )
    
    node = TradingNode(config=config)
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    
    strategy = OrderBookTestStrategy(inst_id_str)
    node.trader.add_strategy(strategy)
    
    node.build()
    loop.run_until_complete(node.run_async())

if __name__ == '__main__':
    print(">>> Starting L2 Order Book Test (Version V4)")
    threading.Thread(target=run_nautilus, daemon=True).start()
    socketio.run(app, host='127.0.0.1', port=5000, allow_unsafe_werkzeug=True)