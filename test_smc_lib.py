import os
import sys

class NullWriter:
    def write(self, text): pass
    def flush(self): pass

old_stdout = sys.stdout
sys.stdout = NullWriter()
try:
    from smartmoneyconcepts import smc
finally:
    sys.stdout = old_stdout

import pandas as pd
from live_packet_builder import fetch_historical_packets, get_adaptive_bucket

def main():
    symbol = "DENTUSDT.BINANCE"
    bucket_usd = get_adaptive_bucket(symbol)
    hist, _ = fetch_historical_packets(symbol, bucket_usd=bucket_usd, days=0.5)
    
    df = pd.DataFrame([{"open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume, "timestamp": p.timestamp} for p in hist])
    
    ohlc = df[['open', 'high', 'low', 'close', 'volume']]
    swing = smc.swing_highs_lows(ohlc, swing_length=50)
    
    ob = smc.ob(ohlc, swing)
    fvg = smc.fvg(ohlc)
    
    print("OB Columns:", ob.columns.tolist())
    print("\nSample OB (last 5 active):")
    print(ob[ob['OB'] != 0].tail())
    
    print("\nFVG Columns:", fvg.columns.tolist())
    print("\nSample FVG (last 5 active):")
    print(fvg[fvg['FVG'] != 0].tail())

if __name__ == '__main__':
    main()
