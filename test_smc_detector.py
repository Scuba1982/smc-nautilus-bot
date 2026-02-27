import os
import sys
import io

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if sys.stdout:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr:
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import time
import pandas as pd
from live_packet_builder import fetch_historical_packets, get_adaptive_bucket
from smartmoneyconcepts import smc

def detect_ote_signal_fixed(df, swing_length=50, lag=3):
    if len(df) < swing_length * 4:
        return None
    recent = df.copy().reset_index(drop=True)
    ohlc = recent[['open', 'high', 'low', 'close']]
    
    swing = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    bos = smc.bos_choch(ohlc, swing, close_break=True)
    
    valid_idx_limit = len(recent) - lag - 1
    if valid_idx_limit < 1:
        return None

    # Ищем последний подтверждённый БЫЧИЙ BOS
    bos_list = bos[bos['BOS'] == 1].index.tolist()
    valid_bos = [i for i in bos_list if i <= valid_idx_limit]

    if not valid_bos:
        return None

    bos_idx = valid_bos[-1]
    
    # Point A = Swing Low перед/на bos_idx (корень импульса)
    swing_lows = swing[swing['HighLow'] == -1].index.tolist()
    sl_before = [i for i in swing_lows if i <= bos_idx]
    
    if not sl_before:
        return None
        
    swing_low_idx = sl_before[-1]
    swing_low = float(recent['low'].iloc[swing_low_idx])
    
    # Point B = АБСОЛЮТНЫЙ ПИК после Swing Low до текущего момента
    highs_since_sl = recent['high'].iloc[swing_low_idx:]
    swing_high = float(highs_since_sl.max())
    swing_high_idx = highs_since_sl.idxmax()
    
    if swing_high <= swing_low:
        return None
        
    range_size = swing_high - swing_low
    if range_size <= 0:
        return None
        
    current_price = float(df['close'].iloc[-1])
    
    # ── ПРОВЕРКА ИНВАЛИДАЦИИ ──
    # Чтобы интерфейс не "спамил" ошибками про сломанные зоны или пропущенные движения,
    # мы отсеиваем их сразу здесь.
    
    # 1. Цена уже закрылась НИЖЕ корня импульса:
    if current_price < swing_low:
        return None 
    # 2. Слишком глубокий пробой (>5%)
    if current_price < swing_low * 0.95:
        return None
        
    return {
        'side': 'BUY',
        'swing_low': swing_low,
        'swing_high': swing_high,
        'current_price': current_price
    }

def run_smc_test():
    print("=== SMC DETECTOR STRESS TEST (V3 FIXED) ===")
    coins = ["DENTUSDT", "DOTUSDT", "FILUSDT", "VIRTUALUSDT", "MORPHOUSDT"]
    
    for sym in coins:
        inst_id = sym + ".BINANCE"
        try:
            bucket = get_adaptive_bucket(inst_id)
            hist, _ = fetch_historical_packets(inst_id, bucket_usd=bucket, days=0.5)
            if len(hist) < 200: continue
            
            df = pd.DataFrame([{"open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume, "timestamp": p.timestamp} for p in hist])
            
            # ТЕСТИРУЕМ НОВУЮ ЛОГИКУ
            signal = detect_ote_signal_fixed(df, swing_length=50)
            
            if not signal:
                print(f"[{sym}] DETECTOR: NO SIGNAL (filtered correctly or no structure)")
            else:
                current_price = df.iloc[-1]['close']
                print(f"\n[{sym}] SIGNAL FOUND!")
                print(f"   Current Price: {current_price:.6f}")
                print(f"   Point A (Low) : {signal['swing_low']:.6f}")
                print(f"   Point B (High): {signal['swing_high']:.6f}")
                print(f"   Range Size    : {signal['swing_high'] - signal['swing_low']:.6f}")
                
                # Check 1: Price is already way above Swing High
                if current_price > signal['swing_high']:
                    pct = (current_price - signal['swing_high']) / signal['swing_high'] * 100
                    print(f"   [BUG TRIGGER] Price is ABOVE target by {pct:.2f}%! (Missed Move)")
                
                # Check 2: Price is already below Swing Low (Structure broken)
                if current_price < signal['swing_low']:
                    print(f"   [BUG TRIGGER] Price is BELOW low! (Structure Broken)")
                    
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")

if __name__ == '__main__':
    run_smc_test()
