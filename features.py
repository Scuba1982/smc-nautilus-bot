import numpy as np
from typing import List

def calculate_packet_metrics(packet_prices: List[float],
                             buy_volume: float,
                             sell_volume: float,
                             volume: float) -> dict:
    """
    Рассчитывает метрики для одного пакета.
    Возвращает словарь с imbalance, vpin, volatility.
    """
    total_vol = buy_volume + sell_volume
    if total_vol == 0:
        imbalance = 0.0
        vpin = 0.0
    else:
        imbalance = (buy_volume - sell_volume) / total_vol
        vpin = abs(buy_volume - sell_volume) / total_vol

    # Волатильность: стандартное отклонение лог-доходностей внутри пакета
    if len(packet_prices) < 2:
        volatility = 0.0001
    else:
        log_returns = [np.log(packet_prices[i] / packet_prices[i-1])
                       for i in range(1, len(packet_prices))]
        volatility = np.std(log_returns, ddof=1) if len(log_returns) > 1 else 0.0001

    return {
        'imbalance': imbalance,
        'vpin': vpin,
        'volatility': volatility
    }

def calculate_er(close_prices: List[float], period: int = 20) -> float:
    """Efficiency Ratio за последние period баров."""
    if len(close_prices) < period + 1:
        return 0.5
    closes = close_prices[-period-1:]
    net_change = abs(closes[-1] - closes[0])
    path = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
    return net_change / path if path > 0 else 0.5