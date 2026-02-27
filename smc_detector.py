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
            ote_high      : bos_high - range_size * 0.62  (62% ретрейс — верхняя граница)
            ote_low       : bos_high - range_size * 0.79  (79% ретрейс — нижняя граница)
            entry_price   : bos_high - range_size * 0.705 (70.5% откат = Sweet Spot)
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

    # Ищем последний подтверждённый БЫЧИЙ BOS
    # Нам не важно, были ли после него мелкие медвежьи сломы, так как откат в зону OTE
    # часто сопровождается локальными медвежьими сломами структуры на минутках.
    bos_list = bos[bos['BOS'] == 1].index.tolist()
    valid_bos = [i for i in bos_list if i <= valid_idx_limit]

    if not valid_bos:
        return None

    bos_idx = valid_bos[-1]

    # ── ПРАВИЛЬНЫЙ DEALING RANGE ─────────────────────────────────────────
    # Point A = Swing Low перед/на bos_idx (корень импульса)
    swing_lows  = swing[swing['HighLow'] == -1].index.tolist()
    sl_before = [i for i in swing_lows if i <= bos_idx]

    if not sl_before:
        # print("[SMC DEBUG] структура не найдена (нет sl_before)")
        return None

    swing_low_idx  = sl_before[-1]
    swing_low  = float(recent['low'].iloc[swing_low_idx])

    # Point B = АБСОЛЮТНЫЙ ПИК после Swing Low до текущего момента
    highs_since_sl = recent['high'].iloc[swing_low_idx:]
    swing_high = float(highs_since_sl.max())
    swing_high_idx = highs_since_sl.idxmax()

    # Если пик оказался самим корнем (невозможно, но мало ли)
    if swing_high <= swing_low:
        return None

    range_size = swing_high - swing_low
    if range_size <= 0:
        # print(f"[SMC DEBUG] Пропуск: range_size <= 0 ({range_size})")
        return None

    # ── ЗОНЫ ФИБОНАЧЧИ (откат сверху вниз — как в backtest) ─────────────
    ote_high      = swing_high - range_size * 0.62
    ote_low       = swing_high - range_size * 0.79
    entry_price   = swing_high - range_size * 0.705

    tp_conservative = swing_high
    tp_standard     = swing_high + range_size * 0.27
    tp_aggressive   = swing_high + range_size * 0.618

    stop_loss       = swing_low * (1 - 0.001)

    current_price = float(df['close'].iloc[-1])

    # ── ПРОВЕРКА ИНВАЛИДАЦИИ СРАЗУ (САЙЛЕНТ) ──
    # Чтобы интерфейс не "спамил" ошибками про сломанные зоны, мы отсеиваем их сразу здесь.
    
    # 1. Цена уже закрылась НА или НИЖЕ корня импульса (настоящий слом структуры)
    #    FIX: было `<`, стало `<=` — если цена вернулась ровно к swing_low, импульс полностью отменён.
    if current_price <= swing_low:
        return None 
    # 2. Слишком глубокий пробой (>5%)
    if current_price < swing_low * 0.95:
        return None

    # 3. FIX: Цена НИЖЕ зоны OTE = зона уже пройдена (pump-dump паттерн)
    #    Если цена упала ниже нижней границы OTE (79% Fib), значит это не ретрейсмент,
    #    а полный разворот. Нет смысла ждать отскок.
    if current_price < ote_low:
        return None

    # ── ФИЛЬТР MITIGATED / STALE ЗОН (ЗАПИЛ) ──
    # Как просил юзер: двигаемся хронологически от хая до текущей цены
    # и смотрим, не была ли зона уже протестирована и отработана.
    recent_highs = recent['high'].iloc[swing_low_idx:]
    actual_swing_high_idx = recent_highs.idxmax()
    
    # Срез пакетов ПОСЛЕ достижения Swing High
    post_high = recent.iloc[actual_swing_high_idx:]
    
    # Ищем момент первого касания точки входа (70.5%)
    entered_mask = post_high['low'] <= entry_price
    if entered_mask.any():
        first_entry_idx = entered_mask.idxmax()

        # Срез пакетов ПОСЛЕ первого касания
        post_entry = recent.iloc[first_entry_idx:-1]  # до, но не включая текущий пакет

        if len(post_entry) > 0:
            max_bounce = post_entry['high'].max()
            min_after_entry = post_entry['low'].min()

            # 1. Зона MITIGATED (Отработана): цена зашла в OTE и вышла вверх выше ote_high.
            #    FIX: Было `max_bounce >= swing_high` (слишком строго — ждали возврат к хаю).
            #    Правильно по SMC: если цена побывала в зоне (ниже entry_price) и
            #    отскочила выше ote_high (верхняя граница OTE = 61.8%) — ликвидность собрана,
            #    зона отработана. Повторный вход = ловушка.
            if max_bounce >= ote_high:
                return None

            # 2. FIX: Зона ПРОБИТА ВНИЗ — цена вошла в OTE и провалилась ниже ote_low
            #    Это значит импульс полностью отменён (pump-dump / полный разворот)
            if min_after_entry < ote_low:
                return None

            # 3. Зона STALE (Протухла): цена застряла в зоне слишком долго.
            #    200 пакетов — разумный тайм-аут (было 50 — слишком мало).
            time_since_entry = len(post_high) - list(post_high.index).index(first_entry_idx)
            if time_since_entry > 200:
                return None

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