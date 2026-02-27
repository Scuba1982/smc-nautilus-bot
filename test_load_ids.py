"""
test_load_ids.py -- Test load_ids with 30+ coins to find which ones fail.
Factory MUST be named 'SandboxLiveExecClientFactory' (hardcoded check in node_builder.py).
"""
import asyncio
import os

# Extended list: typical scanner coins + memecoins + small caps
TEST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "MATICUSDT",
    "UNIUSDT", "SHIBUSDT", "PEPEUSDT", "DENTUSDT", "HOTUSDT",
    "TRXUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT", "OPUSDT",
    "FILUSDT", "ATOMUSDT", "ARBUSDT", "SUIUSDT", "SEIUSDT",
    "INJUSDT", "WLDUSDT", "FETUSDT", "BNBUSDT", "TONUSDT",
    "FLOKIUSDT", "BONKUSDT", "WIFUSDT", "BOMEUSDT", "PUMPUSDT",
    "MIRAUSDT", "BARDUSDT", "1000SATSUSDT", "LUNCUSDT", "GALAUSDT",
]


async def test_many_coins():
    print("="*60)
    print("TEST: load_ids with %d coins" % len(TEST_SYMBOLS))
    print("="*60)
    
    try:
        from nautilus_trader.adapters.binance import BINANCE, BinanceAccountType
        from nautilus_trader.adapters.binance import BinanceDataClientConfig, BinanceLiveDataClientFactory
        from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
        from nautilus_trader.adapters.sandbox.execution import SandboxExecutionClient
        from nautilus_trader.live.factories import LiveExecClientFactory
        from nautilus_trader.config import InstrumentProviderConfig, TradingNodeConfig, LoggingConfig, LiveExecEngineConfig
        from nautilus_trader.live.node import TradingNode
        
        load_ids = frozenset(["%s.BINANCE" % s for s in TEST_SYMBOLS])
        
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        sandbox_ref = [None]
        
        class SandboxLiveExecClientFactory(LiveExecClientFactory):
            @staticmethod
            def create(loop, portfolio, msgbus, cache, clock, config, **kwargs):
                client = SandboxExecutionClient(
                    loop=loop, portfolio=portfolio, msgbus=msgbus,
                    cache=cache, clock=clock, config=config
                )
                sandbox_ref[0] = client
                return client
        
        config = TradingNodeConfig(
            trader_id="TEST-MANY",
            logging=LoggingConfig(log_level="WARN"),
            exec_engine=LiveExecEngineConfig(reconciliation=False),
            data_clients={BINANCE: BinanceDataClientConfig(
                api_key=api_key, api_secret=api_secret,
                account_type=BinanceAccountType.SPOT, testnet=False,
                instrument_provider=InstrumentProviderConfig(load_ids=load_ids),
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
        print("  node.build() OK!")
        
        print("  Starting node.run_async() for 15 seconds...")
        
        async def run_briefly():
            task = asyncio.create_task(node.run_async())
            await asyncio.sleep(15)
            
            from nautilus_trader.model.identifiers import InstrumentId
            
            ok_list = []
            fail_list = []
            
            print("\n  --- Results ---")
            for sym in sorted(TEST_SYMBOLS):
                inst_id = InstrumentId.from_str("%s.BINANCE" % sym)
                inst = node.kernel.cache.instrument(inst_id)
                if inst:
                    ok_list.append(sym)
                    print("  [OK]   %-15s price_prec=%-2d size_prec=%-2d base=%-6s" % (
                        sym, inst.price_precision, inst.size_precision, inst.base_currency))
                else:
                    fail_list.append(sym)
                    print("  [FAIL] %-15s NOT in cache" % sym)
            
            print("\n  --- Summary ---")
            print("  OK:   %d/%d" % (len(ok_list), len(TEST_SYMBOLS)))
            print("  FAIL: %d/%d" % (len(fail_list), len(TEST_SYMBOLS)))
            if fail_list:
                print("  Failed coins: %s" % ", ".join(fail_list))
            
            # Check sandbox exchange
            sandbox = sandbox_ref[0]
            if sandbox and hasattr(sandbox, 'exchange'):
                # Count instruments in exchange
                exchange_instruments = sandbox.exchange.instruments if hasattr(sandbox.exchange, 'instruments') else None
                if exchange_instruments:
                    print("  Exchange instruments: %d" % len(exchange_instruments))
            
            print("\n  Disposing node...")
            node.dispose()
        
        await run_briefly()
        print("  DONE!")
        
    except Exception as e:
        print("  ERROR: %s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_many_coins())
