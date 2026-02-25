"""
SMC Detector — правильный Dealing Range.

КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: бос High/Low больше НЕ берётся из одной свечи.
Диапазон строится от корня импульса (Swing Low) до пика (Swing High),
как в канонической методологии ICT/SMC.
"""
import pandas as pd
import numpy as np
from smartmoneyconcepts import smc


def detect_ote_signal(df: pd.DataFrame, swing_length: int = 50, lag: int = 50) -> dict | None:
    """
    Ищет бычий BOS и строит правильный Dealing Range.

    Возвращает:
        dict с ключами:
            side          : 'BUY'
            price         : текущая цена (в зоне OTE)
            swing_low     : Point A — корень импульса (Invalidation / SL уровень)
            swing_high    : Point B — пик импульса (Liquidity / TP уровень)
            range_size    : swing_high - swing_low
            ote_high      : swing_low + range_size * 0.382  (61.8% ретрейс — верхняя граница)
            ote_low       : swing_low + range_size * 0.210  (79.0% ретрейс — нижняя граница)
            entry_price   : swing_low + range_size * 0.295 (70.5% откат = Sweet Spot)
            tp_conservative: swing_high
            tp_standard   : swing_high + range_size * 0.27
            tp_aggressive : swing_high + range_size * 0.618
            bos_index     : индекс пакета где был BOS (для time_since_bos)
            # Обратная совместимость:
            bos_high / bos_low — то же что swing_high / swing_low
        или None если условия не выполнены.
    """
    if len(df) < swing_length * 4: # Минимум свечей для формирования структуры
        return None

    # Используем весь датафрейм (он уже ограничен 1000 в стратегии)
    recent = df.copy().reset_index(drop=True)
    ohlc   = recent[['open', 'high', 'low', 'close']]

    swing  = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    bos    = smc.bos_choch(ohlc, swing, close_break=True)

    # Индекс с учётом лага (достаточно подтверждённый BOS)
    valid_idx_limit = len(recent) - lag - 1
    if valid_idx_limit < 1:
        return None

    # Ищем последний подтверждённый бычий BOS
    bos_list = bos[bos['BOS'] == 1].index.tolist()
    valid_bos = [i for i in bos_list if i <= valid_idx_limit]

    if not valid_bos:
        return None

    bos_idx = valid_bos[-1]

    # ── ПРАВИЛЬНЫЙ DEALING RANGE ─────────────────────────────────────────
    # Point B = Swing High перед/на bos_idx (пик перед откатом)
    # Ищем последний Swing High ДО(включая) bos_idx
    swing_highs = swing[swing['HighLow'] == 1].index.tolist()
    swing_lows  = swing[swing['HighLow'] == -1].index.tolist()

    # Swing High — последний максимум до и включая bos_idx
    sh_before = [i for i in swing_highs if i <= bos_idx]
    sl_before = [i for i in swing_lows  if i <= bos_idx]

    if not sh_before or not sl_before:
        print("[SMC DEBUG] Fallback удален: структура не найдена (нет sh_before или sl_before)")
        return None

    swing_high_idx = sh_before[-1]
    swing_low_idx  = sl_before[-1]

    swing_high = float(recent['high'].iloc[swing_high_idx])
    swing_low  = float(recent['low'].iloc[swing_low_idx])

    # Swing Low должен быть ДО Swing High (корень импульса)
    if swing_low_idx >= swing_high_idx:
        # Ищем Swing Low ДО Swing High
        sl_before_sh = [i for i in swing_lows if i < swing_high_idx]
        if sl_before_sh:
            swing_low = float(recent['low'].iloc[sl_before_sh[-1]])
        else:
            print("[SMC DEBUG] Структура отменена: swing_low >= swing_high и нет предыдущих swing_lows")
            return None

    range_size = swing_high - swing_low
    if range_size <= 0:
        print(f"[SMC DEBUG] Пропуск: range_size <= 0 ({range_size})")
        return None

    # ── ЗОНЫ ФИБОНАЧЧИ (откат сверху вниз) ──────────────────────────────
    ote_high      = swing_low + range_size * 0.382
    ote_low       = swing_low + range_size * 0.210
    entry_price   = swing_low + range_size * 0.295

    tp_conservative = swing_high
    tp_standard     = swing_high + range_size * 0.27
    tp_aggressive   = swing_high + range_size * 0.618

    stop_loss       = swing_low * (1 - 0.001)

    current_price = float(df['close'].iloc[-1])

    # [ИСПРАВЛЕНИЕ] Мы больше не блокируем выдачу сигнала, если цена еще не в OTE.
    # Детектор просто возвращает структуру (Dealing Range).
    # А стратегия (SMCStrategy) сохранит это как _pending_signal и будет ждать
    # падения цены в зону OTE на живых тиках.

    return {
        'side':            'BUY',
        'price':           current_price,
        # Dealing Range
        'swing_low':       swing_low,
        'swing_high':      swing_high,
        'range_size':      range_size,
        # Fibonacci levels
        'ote_low':         ote_low,
        'ote_high':        ote_high,
        'entry_price':     entry_price,    # 70.5% Sweet Spot
        'stop_loss':       stop_loss,      # структурный SL
        # TP targets
        'tp_conservative': tp_conservative,
        'tp_standard':     tp_standard,
        'tp_aggressive':   tp_aggressive,
        # Обратная совместимость
        'bos_high':        swing_high,
        'bos_low':         swing_low,
        'bos_index':       (len(df) - len(recent)) + bos_idx,
    }