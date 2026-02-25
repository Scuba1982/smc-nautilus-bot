import pandas as pd
from datetime import datetime

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
from decimal import Decimal

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue, TradeId
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.enums import OmsType, AccountType, AggressorSide
from nautilus_trader.model.currencies import Currency
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from smc_nautilus_strategy import MultiSMCStrategy
from live_packet_builder import VolumePacketBuilder
def load_config():
    cfg = {}
    with open('config.yaml', 'r') as f:
        content = f.read()
        
    m_inst = re.search(r'instrument_id:\s*(.+)', content)
    if m_inst:
        cfg['instrument_id'] = m_inst.group(1).strip()
        
    m_model = re.search(r'model_path:\s*(.+)', content)
    if m_model:
        cfg['model_path'] = m_model.group(1).strip()
        
    return cfg

def run_simulation():
    print("=" * 60)
    print("🚀 СИМУЛЯЦИЯ LIVE-ТОРГОВЛИ ЧЕРЕЗ BACKTEST ENGINE")
    print("=" * 60)
    
    cfg = load_config()
    instrument_id_str = cfg.get('instrument_id', 'BTCUSDT.BINANCE')
    instrument_id = InstrumentId.from_str(instrument_id_str)
    model_path = cfg.get('model_path', 'models/smc_model.json')
    
    # 1. Загрузка успешной сделки из датасета
    print("Загрузка исторических пакетов для симуляции...")
    df = pd.read_parquet('data/BTCUSDT_2026_aggTrades_packets.parquet')
    
    start_idx = 0
    end_idx = len(df)
    sub_df = df.iloc[start_idx:end_idx]
    
    # 2. Инициализация Nautilus Backtest Engine
    engine_config = BacktestEngineConfig(trader_id=TraderId("SIMULATOR-001"))
    engine = BacktestEngine(config=engine_config)
    
    usdt = Currency.from_str("USDT")
    engine.add_venue(
        venue=Venue("BINANCE"),
        oms_type=OmsType.HEDGING,
        account_type=AccountType.MARGIN,
        base_currency=usdt,
        starting_balances=[Money(1_000_000, usdt)]
    )
    
    instrument = TestInstrumentProvider.btcusdt_binance()
    engine.add_instrument(instrument)
    
    # 3. Конфигурация Стратегии
    strategy_cfg = {
        'instrument_id': str(instrument_id),
        'swing_length': 50,
        'lag': 50,
        'model_threshold': 0.5,
        'model_path': model_path,
    }
    
    symbols_configs = {
        str(instrument_id): {
            'bucket_usd': 20_000_000.0,
            'hist_packets': [],
            'packet_builder': VolumePacketBuilder(bucket_usd=20_000_000.0)
        }
    }
    
    strategy = MultiSMCStrategy(strategy_config=strategy_cfg, symbols_configs=symbols_configs, gui_queue=None)
    engine.add_strategy(strategy)
    
    print("\nГенерация тиков из пакетов...")
    ticks = []
    tick_count = 0
    
    for idx, row in sub_df.iterrows():
        # Эмуляция 4 тиков на пакет (Open, Low, High, Close)
        ticks_sim = [
            (row['open'], "BUYER"),
            (row['low'], "SELLER"),
            (row['high'], "BUYER"),
            (row['close'], "SELLER" if row['close'] < row['open'] else "BUYER")
        ]
        
        ts = row['timestamp']
        base_ts = ts - 1_000_000_000 # Минус 1 секунда для начала пакета
        
        for i, (price_val, side) in enumerate(ticks_sim):
            aggressor = AggressorSide.BUYER if side == "BUYER" else AggressorSide.SELLER
            
            tick = TradeTick(
                instrument_id=instrument_id,
                price=Price.from_str(f"{price_val:.2f}"),
                size=Quantity.from_str(f"{row['volume'] / 4:.6f}"),
                ts_event=base_ts + (i * 250_000_000), 
                ts_init=base_ts + (i * 250_000_000),
                aggressor_side=aggressor,
                trade_id=TradeId(str(tick_count))
            )
            tick_count += 1
            ticks.append(tick)
            
    engine.add_data(ticks)
    
    print(f"\nЗапуск симуляции на {len(ticks)} тиках...")
    engine.run()
    
    print("\n" + "=" * 60)
    print("✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    
    # Вывод отчета
    report = engine.trader.generate_account_report(Venue("BINANCE"))
    print("\nОТЧЕТ АККАУНТА (БЕКТЕСТ):")
    print(report)
    print("=" * 60)

if __name__ == "__main__":
    run_simulation()
