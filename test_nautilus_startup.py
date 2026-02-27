"""
test_nautilus_startup.py - Diagnostic test for Nautilus TradingNode startup.
Tests the exact phase where the crash happens: AFTER data loading, DURING node startup.
Run: python test_nautilus_startup.py
"""
import asyncio
import sys
import os
import time
import traceback
import queue
import psutil

# Fix encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

def get_memory_mb():
    """Current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

async def run():
    mem_start = get_memory_mb()
    print(f"[MEM] Start: {mem_start:.0f} MB")
    
    # --- Phase 1: Load config ---
    import yaml
    with open("config.yaml", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    
    api_key = CONFIG.get("api_key", "")
    api_secret = CONFIG.get("api_secret", "")
    cfg = CONFIG.get("strategy", {})
    
    # --- Phase 2: Load data for N coins ---
    from live_packet_builder import fetch_historical_packets, get_adaptive_bucket
    
    # Test with exact number of coins that crashes
    coins = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "SHIBUSDT",
        "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "ALGOUSDT",
        "PEPEUSDT", "DENTUSDT", "APEUSDT",  # 18 coins
    ]
    
    N = int(sys.argv[1]) if len(sys.argv) > 1 else len(coins)
    coins = coins[:N]
    print(f"\n[TEST] Loading {len(coins)} coins...")
    
    symbols_configs = {}
    for i, sym in enumerate(coins):
        inst_id = f"{sym}.BINANCE"
        try:
            bucket = get_adaptive_bucket(inst_id)
            hist, pb = fetch_historical_packets(symbol=inst_id, bucket_usd=bucket, days=0.5)
            symbols_configs[inst_id] = {
                'bucket_usd': bucket,
                'hist_packets': hist,
                'packet_builder': pb
            }
            print(f"  [{i+1}/{len(coins)}] {sym}: {len(hist)} packets, bucket=${bucket:,.0f}")
        except Exception as e:
            print(f"  [{i+1}/{len(coins)}] {sym}: SKIP ({e})")
        
        if i > 0 and i % 5 == 0:
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(2)
    
    mem_after_data = get_memory_mb()
    print(f"\n[MEM] After data load: {mem_after_data:.0f} MB (+{mem_after_data - mem_start:.0f} MB)")
    print(f"[TEST] Loaded {len(symbols_configs)}/{len(coins)} coins successfully")
    
    # --- Phase 3: Create Nautilus Node ---
    print(f"\n{'='*60}")
    print(f"[PHASE 3] Creating TradingNode...")
    print(f"{'='*60}")
    
    try:
        from nautilus_trader.adapters.binance import (
            BINANCE, BinanceDataClientConfig, BinanceLiveDataClientFactory,
        )
        from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
        from nautilus_trader.config import TradingNodeConfig, LoggingConfig, LiveExecEngineConfig, InstrumentProviderConfig
        from nautilus_trader.live.node import TradingNode
        
        from nautilus_trader.live.factories import LiveExecClientFactory
        from nautilus_trader.adapters.sandbox.execution import SandboxExecutionClient
        
        # Inline factory — same as in flask_server.py (nested class, can't import)
        class SandboxLiveExecClientFactory(LiveExecClientFactory):
            @staticmethod
            def create(loop, portfolio, msgbus, cache, clock, config, **kwargs):
                return SandboxExecutionClient(
                    loop=loop, portfolio=portfolio, msgbus=msgbus,
                    cache=cache, clock=clock, config=config
                )
        
        account_type = "SPOT" if not CONFIG.get("testnet") else "SPOT_TESTNET"
        
        node_config = TradingNodeConfig(
            trader_id="TEST-DIAG",
            logging=LoggingConfig(log_level="INFO"),
            exec_engine=LiveExecEngineConfig(reconciliation=False),
            data_clients={BINANCE: BinanceDataClientConfig(
                api_key=api_key, api_secret=api_secret,
                account_type=account_type, testnet=False,
                instrument_provider=InstrumentProviderConfig(load_all=False),
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
        print(f"  [OK] TradingNode created")
        
    except Exception as e:
        print(f"  [CRASH] TradingNode creation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    
    mem_after_node = get_memory_mb()
    print(f"[MEM] After node create: {mem_after_node:.0f} MB (+{mem_after_node - mem_after_data:.0f} MB)")
    
    # --- Phase 4: Create Strategy ---
    print(f"\n{'='*60}")
    print(f"[PHASE 4] Creating MultiSMCStrategy with {len(symbols_configs)} instruments...")
    print(f"{'='*60}")
    
    try:
        from smc_nautilus_strategy import MultiSMCStrategy
        
        gui_q = queue.Queue()
        manual_q = queue.Queue()
        
        strategy = MultiSMCStrategy(
            strategy_config=cfg,
            symbols_configs=symbols_configs,
            gui_queue=gui_q,
            manual_order_queue=manual_q,
        )
        print(f"  [OK] Strategy created with {len(strategy.instruments_data)} instruments")
        
    except Exception as e:
        print(f"  [CRASH] Strategy creation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    
    mem_after_strat = get_memory_mb()
    print(f"[MEM] After strategy: {mem_after_strat:.0f} MB (+{mem_after_strat - mem_after_node:.0f} MB)")
    
    # --- Phase 5: Build + Run ---
    print(f"\n{'='*60}")
    print(f"[PHASE 5] node.build() + PRE-REGISTER + node.run_async()...")
    print(f"{'='*60}")
    
    try:
        node.trader.add_strategy(strategy)
        node.build()
        print(f"  [OK] node.build() done")
    except Exception as e:
        print(f"  [CRASH] node.build() failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return
    
    # Pre-register instruments (HOT_SANDBOX_FIX_V2)
    try:
        import requests
        print("  [PRE-REG] Fetching exchangeInfo...")
        resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        weight = resp.headers.get("x-mbx-used-weight-1m", "?")
        print(f"  [PRE-REG] exchangeInfo status={resp.status_code}, used_weight={weight}")
        
        if resp.status_code == 200:
            symbols_info = resp.json().get('symbols', [])
            usdt_pairs = [s['symbol'] for s in symbols_info if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
            print(f"  [PRE-REG] Found {len(usdt_pairs)} USDT pairs to register")
            
            from nautilus_trader.model.instruments import CurrencyPair
            from nautilus_trader.model.currencies import USDT, BTC
            from nautilus_trader.model.identifiers import InstrumentId, Symbol
            from nautilus_trader.model.objects import Price, Quantity
            from decimal import Decimal
            
            added = 0
            for sym_name in usdt_pairs:
                iid = InstrumentId.from_str(f"{sym_name}.BINANCE")
                if not node.kernel.cache.instrument(iid):
                    instrument = CurrencyPair(
                        instrument_id=iid,
                        raw_symbol=Symbol(sym_name),
                        base_currency=BTC,
                        quote_currency=USDT,
                        price_precision=8,
                        size_precision=8,
                        price_increment=Price.from_str("0.00000001"),
                        size_increment=Quantity.from_str("0.00000001"),
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
                    added += 1
            
            print(f"  [PRE-REG] Added {added} instruments to cache")
    except Exception as e:
        print(f"  [PRE-REG] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    mem_after_prereg = get_memory_mb()
    print(f"[MEM] After pre-register: {mem_after_prereg:.0f} MB (+{mem_after_prereg - mem_after_strat:.0f} MB)")
    
    # --- Phase 6: Run (with timeout) ---
    print(f"\n{'='*60}")
    print(f"[PHASE 6] node.run_async() — monitoring for 30 seconds...")
    print(f"{'='*60}")
    
    try:
        # Run node in background
        run_task = asyncio.create_task(node.run_async())
        
        for sec in range(30):
            await asyncio.sleep(1)
            mem_now = get_memory_mb()
            # Check if task crashed
            if run_task.done():
                exc = run_task.exception()
                if exc:
                    print(f"\n  [CRASH] node.run_async() died at t+{sec+1}s!")
                    print(f"  Error: {type(exc).__name__}: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                else:
                    print(f"\n  [STOP] node.run_async() completed normally at t+{sec+1}s")
                return
            
            if sec % 5 == 0:
                print(f"  [ALIVE] t+{sec}s, mem={mem_now:.0f} MB, node state={getattr(node, '_state', '?')}")
        
        print(f"\n  [OK] Survived 30 seconds! Node is stable.")
        
    except Exception as e:
        print(f"\n  [CRASH] Exception during run: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        print(f"\n[MEM] Final: {get_memory_mb():.0f} MB")
        try:
            node.dispose()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(run())
