"""
test_load_all_coins.py -- Use load_all=True to load ALL Binance instruments.
Find every coin that fails due to Nautilus maxQty bug.
"""
import asyncio
import os
import requests


async def test_load_all():
    print("="*60)
    print("TEST: load_all=True (all Binance instruments)")
    print("="*60)
    
    # First get all USDT trading pairs from API
    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
    symbols_info = resp.json().get('symbols', [])
    all_usdt = sorted([s['symbol'] for s in symbols_info 
                       if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'])
    print("  Total USDT trading pairs on Binance: %d" % len(all_usdt))
    
    try:
        from nautilus_trader.adapters.binance import BINANCE, BinanceAccountType
        from nautilus_trader.adapters.binance import BinanceDataClientConfig, BinanceLiveDataClientFactory
        from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
        from nautilus_trader.adapters.sandbox.execution import SandboxExecutionClient
        from nautilus_trader.live.factories import LiveExecClientFactory
        from nautilus_trader.config import InstrumentProviderConfig, TradingNodeConfig, LoggingConfig, LiveExecEngineConfig
        from nautilus_trader.live.node import TradingNode
        
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        class SandboxLiveExecClientFactory(LiveExecClientFactory):
            @staticmethod
            def create(loop, portfolio, msgbus, cache, clock, config, **kwargs):
                return SandboxExecutionClient(
                    loop=loop, portfolio=portfolio, msgbus=msgbus,
                    cache=cache, clock=clock, config=config
                )
        
        config = TradingNodeConfig(
            trader_id="TEST-ALL",
            logging=LoggingConfig(log_level="WARN"),
            exec_engine=LiveExecEngineConfig(reconciliation=False),
            data_clients={BINANCE: BinanceDataClientConfig(
                api_key=api_key, api_secret=api_secret,
                account_type=BinanceAccountType.SPOT, testnet=False,
                instrument_provider=InstrumentProviderConfig(load_all=True),
            )},
            exec_clients={BINANCE: SandboxExecutionClientConfig(
                venue="BINANCE", account_type="MARGIN", oms_type="NETTING",
                base_currency="USDT", starting_balances=["1000000 USDT"],
            )}
        )
        
        node = TradingNode(config=config)
        node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
        node.add_exec_client_factory(BINANCE, SandboxLiveExecClientFactory)
        
        print("  node.build()...")
        node.build()
        print("  Starting node for 25 seconds (loading ALL instruments)...")
        
        async def run_briefly():
            task = asyncio.create_task(node.run_async())
            await asyncio.sleep(25)
            
            from nautilus_trader.model.identifiers import InstrumentId
            
            ok_list = []
            fail_list = []
            
            for sym in all_usdt:
                inst_id = InstrumentId.from_str("%s.BINANCE" % sym)
                inst = node.kernel.cache.instrument(inst_id)
                if inst:
                    ok_list.append(sym)
                else:
                    fail_list.append(sym)
            
            # Total in cache
            all_insts = node.kernel.cache.instruments()
            total_cached = len(all_insts) if all_insts else 0
            
            print("\n  === RESULTS ===")
            print("  Total in cache: %d" % total_cached)
            print("  USDT OK:   %d/%d" % (len(ok_list), len(all_usdt)))
            print("  USDT FAIL: %d/%d" % (len(fail_list), len(all_usdt)))
            
            if fail_list:
                print("\n  --- FAILED USDT COINS ---")
                for sym in fail_list:
                    print("    %s" % sym.encode('ascii', 'replace').decode())
                
                # Save ban list as Python module
                with open("nautilus_banned.py", "w", encoding="utf-8") as f:
                    f.write("# Auto-generated: coins that fail Nautilus InstrumentProvider\n")
                    f.write("# Reason: maxQty out of range [0.0, 18_446_744_073.0]\n")
                    f.write("# Total: %d coins\n" % len(fail_list))
                    f.write("BANNED_COINS = {\n")
                    for sym in sorted(fail_list):
                        f.write('    "%s",\n' % sym)
                    f.write("}\n")
                print("\n  Saved to nautilus_banned.py")
            else:
                print("\n  No failed coins!")
            
            node.dispose()
        
        await run_briefly()
        print("  DONE!")
        
    except Exception as e:
        print("  ERROR: %s: %s" % (type(e).__name__, str(e).encode('ascii', 'replace').decode()))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_load_all())
