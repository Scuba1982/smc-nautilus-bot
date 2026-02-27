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
from smc_detector import detect_ote_signal

def check_mitigation(df, signal):
    swing_high_idx = signal['bos_index'] # Approximation. Wait, bos_index is not swing_high_idx.
    # Let's find swing_high_idx manually based on the price
    swing_high = signal['swing_high']
    # The peak since swing_low.
    recent_highs = df['high']
    # Find the index of swing_high
    # It must equal swing_high
    swing_high_idx = recent_highs[recent_highs == swing_high].index[-1]
    
    post_high = df.iloc[swing_high_idx:]
    
    entry = signal['entry_price']
    ote_high = signal['ote_high']
    
    # Check if price dropped below entry at some point
    entered_mask = post_high['low'] <= entry
    if not entered_mask.any():
        return False, "Not touched yet"
        
    first_entry_idx = entered_mask.idxmax()
    
    # Now check if it bounced significantly after the first entry
    post_entry = df.iloc[first_entry_idx:-1] # exclude current candle
    if len(post_entry) == 0:
        return False, "Just entered"
        
    max_bounce = post_entry['high'].max()
    if max_bounce >= ote_high:
        return True, f"Mitigated (bounced to {max_bounce:.6f} > {ote_high:.6f})"
        
    # Check if it's been chopping around entry for too long (e.g. > 50 packets)
    time_since_entry = len(post_high) - list(post_high.index).index(first_entry_idx)
    if time_since_entry > 50:
         return True, f"Stale (entered {time_since_entry} packets ago and chopped)"
         
    return False, "Valid"

def main():
    coins = ["DENT", "PENGU", "ADA"]
    for sym in coins:
        symbol = sym + "USDT.BINANCE"
        bucket_usd = get_adaptive_bucket(symbol)
        hist, _ = fetch_historical_packets(symbol, bucket_usd=bucket_usd, days=0.5)
        df = pd.DataFrame([{"open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume, "timestamp": p.timestamp} for p in hist])
        
        signal = detect_ote_signal(df)
        if signal:
            is_mit, reason = check_mitigation(df, signal)
            print(f"[{sym}] Signal Found. Mitigated? {is_mit} ({reason})")
        else:
            print(f"[{sym}] No signal")

if __name__ == '__main__':
    main()
