import time
import requests
from live_packet_builder import fetch_historical_packets, get_adaptive_bucket
import sys

# Test only 2 coins to see the crash vs speed
coins = ["BTCUSDT", "SOMIUSDT"]
days_to_fetch = 0.5 # 12 hours for faster test

def on_prog(pct, msg):
    sys.stdout.write(f"\r{msg}")
    sys.stdout.flush()

if __name__ == '__main__':
    for sym in coins:
        print(f"\n\n--- Fetching {sym} ({days_to_fetch} days) ---")
        start = time.time()
        try:
            bucket = get_adaptive_bucket(sym + ".BINANCE")
            hist, pb = fetch_historical_packets(
                symbol=sym + ".BINANCE",
                bucket_usd=bucket,
                days=days_to_fetch,
                on_progress=on_prog
            )
            dur = time.time() - start
            print(f"\n[OK] {sym} loaded {len(hist)} packets in {dur:.1f}s")
        except Exception as e:
            print(f"\n[ERROR] {sym} crashed: {e}")
