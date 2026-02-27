import os
import sys

class NullWriter:
    def write(self, text): pass
    def flush(self): pass

# Suppress the library's stupid print command with emojis
old_stdout = sys.stdout
sys.stdout = NullWriter()
try:
    from smartmoneyconcepts import smc
finally:
    sys.stdout = old_stdout

import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from live_packet_builder import fetch_historical_packets, get_adaptive_bucket
from smc_detector import detect_ote_signal

def plot_smc(sym, df, signal):
    plt.figure(figsize=(14, 8))
    
    # Plot closing prices
    plt.plot(df.index, df['close'], label='Close Price', color='white', linewidth=1)
    
    # Structural Points
    swing_low = signal['swing_low']
    swing_high = signal['swing_high']
    
    plt.axhline(swing_low, color='red', linestyle='--', alpha=0.5, label='Swing Low (0)')
    plt.axhline(swing_high, color='green', linestyle='--', alpha=0.5, label='Swing High (1)')
    
    # OTE Zone (70.5% to 79.0%)
    ote_high = signal['ote_high']
    ote_low = signal['ote_low']
    entry_705 = signal['entry_price']
    
    plt.axhspan(ote_low, ote_high, color='cyan', alpha=0.2, label='OTE Zone')
    plt.axhline(entry_705, color='cyan', linestyle='-', label='Entry (0.705)')
    
    # Draw current price
    current_price = df['close'].iloc[-1]
    plt.scatter([df.index[-1]], [current_price], color='yellow', s=100, zorder=5, label=f'Current: {current_price:.6f}')
    
    # Formatting
    plt.title(f"{sym} SMC Setup (Bucket-based)\nEntry: {entry_705:.6f} | SL: {swing_low:.6f} | TP: {signal['tp_standard']:.6f}")
    plt.xlabel('Packet Index')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    # Dark theme
    plt.gca().set_facecolor('#1e1e1e')
    plt.gcf().patch.set_facecolor('#1e1e1e')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    plt.tick_params(colors='white')
    
    plt.tight_layout()
    # Save chart
    filename = f"{sym}_smc_verification.png"
    plt.savefig(filename, dpi=150, facecolor='#1e1e1e')
    print(f"[{sym}] Chart saved to {filename}")
    plt.close()

def main():
    coins = ["DENT", "PENGU", "ADA"]
    for sym in coins:
        symbol = sym + "USDT.BINANCE"
        print(f"\n--- Verifying {sym}USDT ---")
        try:
            bucket_usd = get_adaptive_bucket(symbol)
            print(f"Bucket: {bucket_usd}")
            hist, _ = fetch_historical_packets(symbol, bucket_usd=bucket_usd, days=0.5)
            
            if len(hist) < 200:
                print(f"Not enough packets: {len(hist)}")
                continue
                
            df = pd.DataFrame([{
                "open": p.open, "high": p.high, "low": p.low, "close": p.close, 
                "volume": p.volume, "timestamp": p.timestamp
            } for p in hist])
            
            print(f"Dataframes shape: {df.shape}")
            signal = detect_ote_signal(df, swing_length=50) # Assuming this is the strategy's length
            
            if signal:
                print(f"[{sym}] SIGNAL FOUND!")
                print(f"  Entry: {signal['entry_price']:.6f}")
                print(f"  OTE: {signal['ote_low']:.6f} - {signal['ote_high']:.6f}")
                print(f"  Current Price: {df['close'].iloc[-1]:.6f}")
                plot_smc(sym, df, signal)
            else:
                print(f"[{sym}] NO SIGNAL detected by current logic.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
