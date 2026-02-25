import time as _time
import numpy as np
import requests
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from packet_builder_numba import process_trades_numba

PACKETS_PER_DAY = 1500  # целевое кол-во пакетов в сутки


def get_adaptive_bucket(symbol: str, testnet: bool = False) -> float:
    """
    Рассчитывает bucket_usd из суточного объёма Binance (один HTTP запрос).

    Формула: bucket_usd = quoteVolume_24h / 1500
    ВАЖНО: объём всегда берётся с mainnet Binance API — на testnet нет реального объёма.

    Примеры:
        BTC  ~$30B/day  -> bucket ~$20M
        SOL  ~$3B/day   -> bucket ~$2M
        DOGE ~$500M/day -> bucket ~$330k
    """
    try:
        raw_symbol = symbol.replace(".BINANCE", "").replace("-PERP", "")
        # Объём ВСЕГДА с mainnet — у testnet нет реальных данных
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": raw_symbol},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        quote_volume = float(data["quoteVolume"])
        if quote_volume < 1_000_000:
            print(f"[BUCKET] {raw_symbol}: volume too low (${quote_volume:,.0f}), using fallback")
            return 5000.0
        bucket = max(quote_volume / PACKETS_PER_DAY, 100.0)
        print(f"[BUCKET] {raw_symbol}: 24h vol=${quote_volume/1e6:.1f}M -> bucket=${bucket:,.0f}")
        return bucket
    except Exception as e:
        fallback = 5000.0
        print(f"[BUCKET] Warning: {e}. Using fallback=${fallback:,.0f}")
        return fallback


@dataclass
class Packet:
    timestamp: int
    end_timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    volume_usd: float
    buy_volume: float
    sell_volume: float
    trades_count: int
    prices: List[float]
    vpin: float = 0.0

    @property
    def imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        return (self.buy_volume - self.sell_volume) / total if total > 0 else 0.0

    @property
    def candle_range(self) -> float:
        """(high - low) / open \u2014 \u043f\u0440\u0438\u0437\u043d\u0430\u043a XGBoost, \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043d\u0430 \u0437\u0430\u043a\u0440\u044b\u0442\u043e\u043c \u043f\u0430\u043a\u0435\u0442\u0435."""
        return (self.high - self.low) / self.open if self.open > 0 else 0.0



@dataclass
class PartialPacket:
    """Текущий незакрытый пакет — обновляется на каждой сделке для GUI."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    buy_volume: float
    sell_volume: float
    volume_usd: float
    bucket_usd: float
    trades_count: int
    prices: List[float]
    _vpin_value: float = 0.0

    @property
    def fill_pct(self) -> float:
        return min(self.volume_usd / self.bucket_usd, 1.0) if self.bucket_usd > 0 else 0.0

    @property
    def imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        return (self.buy_volume - self.sell_volume) / total if total > 0 else 0.0

    @property
    def vpin(self) -> float:
        return self._vpin_value

    @property
    def volatility(self) -> float:
        if len(self.prices) < 2:
            return 0.0001
        log_ret = [np.log(self.prices[i] / self.prices[i - 1])
                   for i in range(1, len(self.prices))]
        return float(np.std(log_ret, ddof=1)) if len(log_ret) > 1 else 0.0001

    @property
    def candle_range(self) -> float:
        """Размер свечи: (high - low) / open. Используется XGBoost как признак."""
        return (self.high - self.low) / self.open if self.open > 0 else 0.0




class VolumePacketBuilder:
    """
    Volume Clock с фиксированным bucket (рассчитанным заранее из 24h API).

    update() возвращает (List[Packet], PartialPacket):
      - List[Packet]   — закрытые пакеты (обычно 0-1, редко 2+ при большой сделке)
      - PartialPacket  — текущий незакрытый пакет с живыми метриками для GUI
    """

    def __init__(self, bucket_usd: float, vpin_window: int = 50):
        self._bucket_usd = bucket_usd
        self._vpin_window = vpin_window
        from collections import deque
        self._history_buy_vols = deque(maxlen=vpin_window)
        self._history_sell_vols = deque(maxlen=vpin_window)
        self._reset()

    def _reset(self):
        self._start_time: Optional[int] = None
        self._volume_usd: float = 0.0
        self._buy_vol: float = 0.0
        self._sell_vol: float = 0.0
        self._trades: int = 0
        self._open: float = 0.0
        self._high: float = -1e18
        self._low: float = 1e18
        self._close: float = 0.0
        self._prices: List[float] = []

    @property
    def bucket_usd(self) -> float:
        return self._bucket_usd

    def update(self, price: float, qty: float, is_sell: bool,
               timestamp_ns: int, trades: int = 1) -> Tuple[List[Packet], PartialPacket]:
        bucket = self._bucket_usd
        usd = price * qty

        if self._start_time is None:
            self._start_time = timestamp_ns
            self._open = price
            self._high = price
            self._low = price

        self._close = price
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        if is_sell:
            self._sell_vol += qty
        else:
            self._buy_vol += qty
        self._volume_usd += usd
        if qty > 0:
            self._trades += trades
        self._prices.append(price)

        # while-петля: большая сделка может породить несколько пакетов
        closed: List[Packet] = []
        while self._volume_usd >= bucket:
            ratio = bucket / self._volume_usd
            pkt_buy = self._buy_vol * ratio
            pkt_sell = self._sell_vol * ratio
            closed.append(Packet(
                timestamp=self._start_time,
                end_timestamp=timestamp_ns,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                volume=pkt_buy + pkt_sell,
                volume_usd=bucket,
                buy_volume=pkt_buy,
                sell_volume=pkt_sell,
                trades_count=self._trades,
                prices=self._prices.copy(),
                vpin=self._calculate_current_vpin(pkt_buy, pkt_sell)
            ))
            
            # Сохраняем в историю для будущих vpin
            self._history_buy_vols.append(pkt_buy)
            self._history_sell_vols.append(pkt_sell)
            
            leftover_usd = self._volume_usd - bucket
            leftover_buy = self._buy_vol * (1.0 - ratio)
            leftover_sell = self._sell_vol * (1.0 - ratio)
            self._reset()
            if leftover_usd > 0:
                self._start_time = timestamp_ns
                self._open = price
                self._high = price
                self._low = price
                self._close = price
                self._buy_vol = leftover_buy
                self._sell_vol = leftover_sell
                self._volume_usd = leftover_usd
                self._prices = [price]

        partial = PartialPacket(
            timestamp=self._start_time or timestamp_ns,
            open=self._open if self._open else price,
            high=self._high if self._high > -1e17 else price,
            low=self._low if self._low < 1e17 else price,
            close=self._close if self._close else price,
            buy_volume=self._buy_vol,
            sell_volume=self._sell_vol,
            volume_usd=self._volume_usd,
            bucket_usd=bucket,
            trades_count=self._trades,
            prices=self._prices.copy(),
            _vpin_value=self._calculate_current_vpin(self._buy_vol, self._sell_vol),
        )
        return closed, partial
        
    def _calculate_current_vpin(self, current_buy: float, current_sell: float) -> float:
        """Считает VPIN за окно _vpin_window включая текущий формирующийся пакет"""
        sum_buy = sum(self._history_buy_vols) + current_buy
        sum_sell = sum(self._history_sell_vols) + current_sell
        total = sum_buy + sum_sell
        if total == 0:
            return 0.0
        return abs(sum_buy - sum_sell) / total


def fetch_historical_packets(
    symbol: str,
    bucket_usd: float,
    days: float = 0.5, # 12 часов истории
    testnet: bool = False,
    on_progress=None,
) -> Tuple[List[Packet], VolumePacketBuilder]:
    """
    Скачивает aggTrades за последние `days` дней через Binance REST API
    и прогоняет их через VolumePacketBuilder.
    Используется пагинация по fromId для максимальной скорости загрузки точных сделок.
    """
    raw_symbol = symbol.replace(".BINANCE", "").replace("-PERP", "")
    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - int(days * 24 * 3600 * 1000)
    total_ms = now_ms - start_ms

    builder = VolumePacketBuilder(bucket_usd=bucket_usd)
    
    prices_list = []
    quantities_list = []
    is_buyer_maker_list = []
    timestamps_list = []
    
    base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
    session = requests.Session()

    print(f"[HISTORY] Fetching {days}d of {raw_symbol} aggTrades by timestamp...")
    if on_progress:
        on_progress(0.0, f"Получение agg Trades для {raw_symbol}...")

    current_ms = start_ms
    fetched_trades = 0
    last_processed_id = -1

    while current_ms < now_ms:
        end_time = min(current_ms + 3600_000, now_ms)
        try:
            resp = session.get(
                f"{base_url}/api/v3/aggTrades",
                params={
                    "symbol": raw_symbol,
                    "startTime": current_ms,
                    "endTime": end_time,
                    "limit": 1000,
                },
                timeout=10,
            )
            resp.raise_for_status()
            trades = resp.json()
        except Exception as e:
            print(f"[HISTORY] Warning: {e}. Retrying in 2s...")
            _time.sleep(2.0)
            continue

        if not trades:
            current_ms = end_time + 1
            continue

        # Собираем данные в сырые списки для быстрой сборки numpy массивов
        for t in trades:
            trade_id = int(t["a"])
            if trade_id <= last_processed_id:
                continue

            ts_ms = int(t["T"])
            if ts_ms <= now_ms:
                prices_list.append(float(t["p"]))
                quantities_list.append(float(t["q"]))
                is_buyer_maker_list.append(bool(t["m"]))
                timestamps_list.append(ts_ms * 1_000_000)
                
            last_processed_id = trade_id

        fetched_trades += len(trades)
        
        # Если уперлись в лимит 1000, значит внутри этого окна еще есть сделки.
        # "Склеиваем" по времени последней сделки.
        if len(trades) == 1000:
            current_ms = int(trades[-1]["T"])
        else:
            current_ms = end_time + 1

        elapsed_ms = current_ms - start_ms
        pct = min(100.0, max(0.0, elapsed_ms / total_ms * 100))
        if on_progress:
            on_progress(pct, f"Загрузка: {fetched_trades} сделок ({pct:.0f}%)")

        _time.sleep(0.04)

    print(f"[HISTORY] Fetch Done: {fetched_trades} trades. Compiling packets via Numba...")
    if on_progress:
        on_progress(100.0, f"Сборка пакетов (Numba)...")

    # 🚀 NUMBA ACCELERATION 🚀
    packets = []
    if fetched_trades > 0:
        prices = np.array(prices_list, dtype=np.float64)
        quantities = np.array(quantities_list, dtype=np.float64)
        timestamps = np.array(timestamps_list, dtype=np.int64)
        is_buyer_maker = np.array(is_buyer_maker_list, dtype=np.bool_)

        res = process_trades_numba(prices, quantities, timestamps, is_buyer_maker, bucket_usd)
        
        o_start, o_end, o_open, o_high, o_low, o_close, o_vol, o_usd, o_buy, o_sell, o_cnt = res
        
        for i in range(len(o_start)):
            vpin_val = builder._calculate_current_vpin(o_buy[i], o_sell[i])
            builder._history_buy_vols.append(o_buy[i])
            builder._history_sell_vols.append(o_sell[i])
            
            p = Packet(
                timestamp=o_start[i],
                end_timestamp=o_end[i],
                open=o_open[i],
                high=o_high[i],
                low=o_low[i],
                close=o_close[i],
                volume=o_vol[i],
                volume_usd=o_usd[i],
                buy_volume=o_buy[i],
                sell_volume=o_sell[i],
                trades_count=o_cnt[i],
                prices=[],
                vpin=vpin_val
            )
            packets.append(p)

    print(f"[HISTORY] Done: {fetched_trades} trades → {len(packets)} packets")
    if on_progress:
        on_progress(100.0, f"History loaded: {len(packets)} packets")
    return packets, builder
