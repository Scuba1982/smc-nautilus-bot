# -*- coding: utf-8 -*-
"""
SEISMIC Historical Data Downloader
===================================
Скачивает aggTrades с data.binance.vision за указанный период.

Usage:
    python download_data.py --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

AGGTRADES_COLUMNS = [
    'agg_trade_id', 'price', 'quantity', 'first_trade_id', 
    'last_trade_id', 'timestamp', 'is_buyer_maker', 'best_price_match'
]


def download_file(url: str, dest_path: str) -> bool:
    """Скачать файл по URL."""
    try:
        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> str:
    """Распаковать ZIP и вернуть путь к CSV."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = zf.namelist()[0]
        zf.extractall(extract_dir)
        return os.path.join(extract_dir, csv_name)


def download_day(symbol: str, date: datetime, temp_dir: str) -> pd.DataFrame:
    """Скачать данные за один день."""
    date_str = date.strftime('%Y-%m-%d')
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{filename}"
    zip_path = os.path.join(temp_dir, filename)
    
    if not download_file(url, zip_path):
        return None
    
    try:
        csv_path = extract_zip(zip_path, temp_dir)
        df = pd.read_csv(csv_path, header=None, names=AGGTRADES_COLUMNS)
        os.remove(csv_path)
        os.remove(zip_path)
        return df
    except Exception as e:
        print(f"Error processing {date_str}: {e}")
        return None


def download_range(symbol: str, start_date: str, end_date: str, max_workers: int = 5):
    """
    Скачать данные за диапазон дат.
    
    Args:
        symbol: Торговая пара (например BTCUSDT)
        start_date: Начальная дата (YYYY-MM-DD)
        end_date: Конечная дата (YYYY-MM-DD)
        max_workers: Количество параллельных загрузок
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Создаём директории
    temp_dir = os.path.join(DATA_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Генерируем список дат
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    
    print(f"[INFO] Downloading {len(dates)} days of {symbol} data...")
    print(f"[INFO] Date range: {start_date} to {end_date}")
    
    all_dfs = []
    failed_dates = []
    
    # Параллельное скачивание
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_day, symbol, d, temp_dir): d for d in dates}
        
        for future in tqdm(as_completed(futures), total=len(dates), desc="Downloading"):
            date = futures[future]
            try:
                df = future.result()
                if df is not None and len(df) > 0:
                    all_dfs.append(df)
                else:
                    failed_dates.append(date.strftime('%Y-%m-%d'))
            except Exception as e:
                failed_dates.append(date.strftime('%Y-%m-%d'))
    
    if not all_dfs:
        print("[ERROR] No data downloaded!")
        return None
    
    # Объединяем все данные
    print("[INFO] Merging data...")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    # Конвертируем типы
    combined['price'] = combined['price'].astype(float)
    combined['quantity'] = combined['quantity'].astype(float)
    combined['timestamp'] = combined['timestamp'].astype(int)
    combined['is_buyer_maker'] = combined['is_buyer_maker'].astype(bool)
    
    # Сохраняем в Parquet (быстрее чем CSV)
    year_range = f"{start.year}" if start.year == end.year else f"{start.year}_{end.year}"
    output_path = os.path.join(DATA_DIR, f"{symbol}_{year_range}_aggTrades.parquet")
    combined.to_parquet(output_path, index=False)
    
    # Статистика
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Symbol:        {symbol}")
    print(f"Total trades:  {len(combined):,}")
    print(f"Date range:    {start_date} to {end_date}")
    print(f"File size:     {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"Output:        {output_path}")
    
    if failed_dates:
        print(f"\n[WARNING] Failed dates ({len(failed_dates)}):")
        for d in failed_dates[:10]:
            print(f"  - {d}")
        if len(failed_dates) > 10:
            print(f"  ... and {len(failed_dates) - 10} more")
    
    # Cleanup temp
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download Binance historical data')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=5, help='Parallel downloads')
    
    args = parser.parse_args()
    
    download_range(args.symbol, args.start, args.end, args.workers)


if __name__ == "__main__":
    main()
