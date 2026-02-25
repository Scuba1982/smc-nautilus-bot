import pandas as pd
import numpy as np
import numba
from numba import jit, float64, int64
import os
import time

# ==========================================
# 🚀 NUMBA OPTIMIZED KERNEL
# ==========================================

@jit(nopython=True)
def calc_trade_imbalance(buy_vol, sell_vol):
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return ((buy_vol - sell_vol) / total) * 100

@jit(nopython=True)
def process_trades_numba(
    prices,
    quantities,
    timestamps,
    is_buyer_maker,
    bucket_usd
):
    """
    Core loop compiled to machine code.
    Correctly splits large trades into multiple fixed‑USD packets.
    """
    n = len(prices)

    # Выходные списки
    out_start_time = []
    out_end_time   = []
    out_open       = []
    out_high       = []
    out_low        = []
    out_close      = []
    out_volume     = []
    out_volume_usd = []
    out_buy_vol    = []
    out_sell_vol   = []
    out_count      = []

    # Состояние текущего (незавершённого) пакета
    current_start_time = -1
    current_buy_vol    = 0.0
    current_sell_vol   = 0.0
    current_total_usd  = 0.0
    current_count      = 0

    p_open  = 0.0
    p_high  = -1.0
    p_low   = 1e18
    p_close = 0.0

    for i in range(n):
        p = prices[i]
        q = quantities[i]
        t = timestamps[i]
        maker = is_buyer_maker[i]

        usd = p * q

        # Инициализация нового пакета, если мы не в сделке
        if current_start_time == -1:
            current_start_time = t
            p_open = p
            p_high = p
            p_low  = p

        # Добавляем данные текущей сделки
        current_total_usd += usd
        current_count += 1
        p_close = p
        if p > p_high:
            p_high = p
        if p < p_low:
            p_low = p

        if maker:
            current_sell_vol += q
        else:
            current_buy_vol += q

        # Пока накопленный объём >= bucket_usd – эмитируем пакеты
        while current_total_usd >= bucket_usd:
            # ---- Сохраняем текущий пакет ----
            out_start_time.append(current_start_time)
            out_end_time.append(t)                # время последней сделки в пакете
            out_open.append(p_open)
            out_high.append(p_high)
            out_low.append(p_low)
            out_close.append(p_close)

            # Объём (количество монет) в этом пакете
            packet_volume = bucket_usd / p
            out_volume.append(packet_volume)
            out_volume_usd.append(bucket_usd)

            # Пропорционально распределяем buy/sell объём
            total_vol = current_buy_vol + current_sell_vol
            if total_vol > 0:
                buy_frac  = current_buy_vol / total_vol
                sell_frac = current_sell_vol / total_vol
            else:
                buy_frac = sell_frac = 0.0

            out_buy_vol.append(packet_volume * buy_frac)
            out_sell_vol.append(packet_volume * sell_frac)
            out_count.append(current_count)       # количество сделок в пакете

            # ---- Вычитаем использованный объём ----
            current_total_usd -= bucket_usd

            # Пропорционально уменьшаем объёмы buy/sell
            current_buy_vol  -= packet_volume * buy_frac
            current_sell_vol -= packet_volume * sell_frac

            # ---- Сбрасываем ценовые метрики для *нового* пакета ----
            # Новый пакет начинается прямо сейчас с текущей сделки (остаток)
            current_start_time = t
            p_open = p
            p_high = p
            p_low  = p
            p_close = p

            # Количество сделок в новом пакете – пока 0 (остаток этой сделки будет добавлен ниже)
            current_count = 0

        # Конец while – остаток сделки остаётся в аккумуляторах для следующего пакета

    return (out_start_time, out_end_time, out_open, out_high, out_low, out_close,
            out_volume, out_volume_usd, out_buy_vol, out_sell_vol, out_count)


# ==========================================
# PANDAS WRAPPER
# ==========================================
def build_packets_numba(file_path, bucket_usd=None):
    print(f"[INFO] Loading {file_path}...")
    df = pd.read_parquet(file_path)
    
    print("[INFO] Normalizing timestamps...")
    # Some rows are in ms, some in us. Let's fix this.
    # We want everything in microseconds for Numba.
    ts_arr = df['timestamp'].values
    mask_ms = ts_arr < 1e14
    ts_arr[mask_ms] *= 1000
    df['timestamp'] = ts_arr
    
    # Auto-adjust bucket if needed
    if bucket_usd is None:
        duration_hours = (df['timestamp'].max() - df['timestamp'].min()) / 1000000 / 3600
        total_vol = (df['price'] * df['quantity']).sum()
        est_daily_vol = total_vol * (24 / (duration_hours + 1e-9))
        bucket_usd = est_daily_vol / 1500
        print(f"[INFO] Adaptive Bucket: ${bucket_usd:,.0f} (Duration: {duration_hours:.2f}h)")
    
    print("[INFO] Preparing arrays for Numba...")
    prices = df['price'].values
    quantities = df['quantity'].values
    timestamps = df['timestamp'].values
    
    is_buyer_maker = df['is_buyer_maker'].values
    
    print("[INFO] Running Numba Kernel...")
    t0 = time.time()
    res = process_trades_numba(prices, quantities, timestamps, is_buyer_maker, bucket_usd)
    dt = time.time() - t0
    print(f"[SUCCESS] Kernel finished in {dt:.2f}s")
    
    # Unpack
    print("[INFO] Constructing DataFrame...")
    out_df = pd.DataFrame({
        'timestamp': res[0],
        'end_timestamp': res[1],
        'open': res[2],
        'high': res[3],
        'low': res[4],
        'close': res[5],
        'volume': res[6],
        'volume_usd': res[7],
        'buy_volume': res[8],
        'sell_volume': res[9],
        'trades_count': res[10]
    })
    
    # Post-process usage metrics that are fast in vectorized pandas
    out_df['duration_sec'] = (out_df['end_timestamp'] - out_df['timestamp']) / 1e6
    # ==============================================================================
    # 🔧 PROFESSIONAL HFT METRICS (FIXED)
    # ==============================================================================
    
    # 1. HFT VOLATILITY (ПРАВИЛЬНАЯ: Размах High-Low)
    # Вместо того чтобы смотреть на закрытия, смотрим на "длину свечи"
    # Это покажет реальный риск выбивания стопа.
    out_df['candle_range'] = (out_df['high'] - out_df['low']) / out_df['open']
    
    # Базовая волатильность = Средний размах за 50 пакетов
    # Это будет наш "мерило" для Адаптивного Стопа
    out_df['volatility'] = out_df['candle_range'].rolling(window=50).mean()
    
    # Если вдруг волатильность 0 (рынок умер), ставим микро-значение, чтобы не ломать деление
    out_df['volatility'] = out_df['volatility'].replace(0, 0.0001)

    # 2. VOL SPIKE (Всплеск волатильности)
    # Показывает, во сколько раз текущая свеча больше средней
    # Если > 3.0 -> Значит сейчас аномалия (надо быть осторожным или ловить нож)
    out_df['vol_spike'] = out_df['candle_range'] / (out_df['volatility'] + 1e-9)

    # 3. IMBALANCE & VPIN (Лента сделок)
    # Защита от деления на ноль, если volume = 0
    vol_safe = out_df['volume'].replace(0, 1.0) # Заменяем 0 на 1, чтобы не делить на ноль
    
    # Чистый дисбаланс (-1 продавцы ... +1 покупатели)
    out_df['imbalance_current'] = (out_df['buy_volume'] - out_df['sell_volume']) / vol_safe
    
    # Токсичность потока (VPIN) - скользящее среднее за 50 пакетов
    # Это настоящий VPIN: сглаживает всплески токсичности через окно 50 пакетов
    vpin_1pkt = (out_df['buy_volume'] - out_df['sell_volume']).abs() / vol_safe
    out_df['vpin'] = vpin_1pkt.rolling(window=50).mean()

    # 4. EFFICIENCY RATIO (ER) - Качество тренда
    # (High - Low) может быть шумным, лучше брать Abs(Change)
    # Эффективность = (Смещение цены) / (Сумма движений)
    period_er = 10
    change = (out_df['close'] - out_df['close'].shift(period_er)).abs()
    path = out_df['close'].diff().abs().rolling(period_er).sum().replace(0, 1e-9)
    out_df['er'] = change / path
    
    # Удаляем NaN, которые появились из-за rolling(50)
    out_df.dropna(inplace=True)
    
    output_file = file_path.replace('.parquet', '_packets.parquet')
    out_df.to_parquet(output_file)
    print(f"[SAVED] {output_file}")
    
if __name__ == "__main__":
    import sys
    bucket = None
    file_path = "data/SOLUSDT_2024_aggTrades.parquet"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    if len(sys.argv) > 2:
        bucket = float(sys.argv[2])
        
    if os.path.exists(file_path):
        build_packets_numba(file_path, bucket_usd=bucket)
    else:
        print(f"[ERROR] File not found: {file_path}")