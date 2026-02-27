import requests
import time
from datetime import datetime

base_url = "https://fapi.binance.com"  # Using Futures API, or "https://api.binance.com" for Spot

def test_limits():
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", 
        "DOGEUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT", "SHIBUSDT", "LTCUSDT",
        "TRXUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT", "XLMUSDT", "TONUSDT",
        "BCHUSDT", "ETCUSDT", "FILUSDT", "ICPUSDT", "VETUSDT", "NEARUSDT"
    ]
    
    session = requests.Session()
    
    print("Testing Binance 1M Weight Limits...\n")
    
    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Fetching 1000 aggTrades for {sym}...")
        
        now_ms = int(time.time() * 1000)
        # Fetch 12 hours of data (roughly 12-24 requests per coin depending on density)
        start_ms = now_ms - (12 * 3600 * 1000)
        current_ms = start_ms
        
        requests_for_coin = 0
        while current_ms < now_ms:
            params = {
                "symbol": sym,
                "limit": 1000,
                "startTime": current_ms,
                "endTime": min(current_ms + 3600_000, now_ms)
            }
            
            resp = session.get(f"https://api.binance.com/api/v3/aggTrades", params=params, timeout=10)
            requests_for_coin += 1
            
            used_weight = int(resp.headers.get("x-mbx-used-weight-1m", 0))
            
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text}")
                return
                
            trades = resp.json()
            if not trades:
                current_ms += 3600_000
                continue
                
            last_ts = int(trades[-1]["T"])
            if last_ts > now_ms:
                break
                
            current_ms = last_ts
            if len(trades) < 1000:
                current_ms += 1
                
        print(f"  Done {sym}. Requests: {requests_for_coin}. Used Weight 1M: {used_weight}/6000")
        
        if used_weight > 5000:
            print("\nWARNING: APPROACHING LIMIT!")
            
if __name__ == '__main__':
    test_limits()
