import asyncio
from flask_server import _FlaskNautilusWorker, CONFIG
from smc_nautilus_strategy import MultiSMCStrategy
import queue

async def run():
    print("Running crash test...")
    CONFIG['testnet'] = False
    gui_queue = queue.Queue()
    worker = _FlaskNautilusWorker(CONFIG, ('BTCUSDT',), gui_queue)
    
    # Overwrite historical loader to return empty so it connects instantly
    import live_packet_builder
    live_packet_builder.fetch_historical_packets = lambda *args, **kwargs: ([], None)
    
    try:
        await worker._run_async()
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())
