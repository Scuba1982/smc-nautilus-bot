"""
test_batch_loading.py - Diagnostic test for batch coin loading.
Emulates what flask_server does when loading 18+ coins.
Run: python test_batch_loading.py
"""
import time
import sys
import traceback
from live_packet_builder import fetch_historical_packets, get_adaptive_bucket

# Typical batch of 18+ coins that causes crash
TEST_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT",
    "SHIBUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
    "FTMUSDT", "ALGOUSDT", "PEPEUSDT",  # <-- coin 18, allegedly crashes
    "DENTUSDT", "APEUSDT", "GALAUSDT",
]

def on_progress(pct, msg):
    sys.stdout.write(f"\r  Progress: {msg} ({pct:.0f}%)   ")
    sys.stdout.flush()

if __name__ == "__main__":
    print(f"=" * 60)
    print(f"BATCH LOADING TEST - {len(TEST_COINS)} coins")
    print(f"=" * 60)
    
    results = []
    total_start = time.time()
    
    for i, sym in enumerate(TEST_COINS):
        inst_id = f"{sym}.BINANCE"
        print(f"\n[{i+1}/{len(TEST_COINS)}] {sym}")
        t0 = time.time()
        
        try:
            # Step 1: Adaptive bucket
            print(f"  Calculating bucket...")
            bucket = get_adaptive_bucket(inst_id)
            print(f"  Bucket: ${bucket:,.0f}")
            
            # Step 2: Historical packets (shorter period for faster test)
            print(f"  Loading history (0.5 day)...")
            hist, pb = fetch_historical_packets(
                symbol=inst_id,
                bucket_usd=bucket,
                days=0.5,  # 12 hours - enough to test, faster than 24h
                on_progress=on_progress
            )
            
            elapsed = time.time() - t0
            print(f"\n  OK: {len(hist)} packets in {elapsed:.1f}s")
            results.append((sym, "OK", len(hist), elapsed))
            
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  CRASH on {sym} after {elapsed:.1f}s!")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results.append((sym, f"CRASH: {type(e).__name__}: {e}", 0, elapsed))
        
        # Same delay pattern as flask_server (BUG 1 FIX)
        if i > 0 and i % 5 == 0:
            print(f"  [COOLDOWN] 10s pause (reduced from 60 for test)")
            time.sleep(10)
        else:
            time.sleep(3)
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"RESULTS (total: {total_elapsed:.0f}s)")
    print(f"{'=' * 60}")
    
    ok_count = 0
    fail_count = 0
    for sym, status, packets, elapsed in results:
        icon = "OK" if status == "OK" else "FAIL"
        if status == "OK":
            ok_count += 1
            print(f"  {icon}  {sym:12s} - {packets:4d} packets ({elapsed:.1f}s)")
        else:
            fail_count += 1
            print(f"  {icon} {sym:12s} - {status}")
    
    print(f"\nTotal: {ok_count} OK, {fail_count} FAIL")
    
    if fail_count > 0:
        sys.exit(1)
