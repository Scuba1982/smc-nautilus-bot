# Инструкция по запуску бектеста в Nautilus Trader

Для скачивания данных и прогона их через бектест Nautilus Trader необходимо выполнить три последовательных шага.

## 1. Скачивание сырых данных (aggTrades)
Используйте скрипт `download_data.py`. Он скачивает ежедневные архивы с `data.binance.vision`.

**Пример команды:**
```bash
python download_data.py --symbol BTCUSDT --start 2026-01-01 --end 2026-02-20
```
*Результат: файл `data/BTCUSDT_2026_aggTrades.parquet`*

---

## 2. Формирование вольюм-пакетов
Для работы стратегии SMC необходимо преобразовать тики в пакеты фиксированного объема с расчетом HFT-метрик (VPIN, Imbalance). Используйте `packet_builder_numba.py`.

**Пример команды:**
```bash
python packet_builder_numba.py data/BTCUSDT_2026_aggTrades.parquet
```
*Результат: файл `data/BTCUSDT_2026_aggTrades_packets.parquet`*

---

## 3. Запуск бектеста через Nautilus
Для прогона стратегии через официальный движок Nautilus Trader используйте `live_simulator.py`. Этот скрипт инициализирует `BacktestEngine`, загружает пакеты и имитирует тиковую торговлю.

**Запуск:**
```bash
python live_simulator.py
```
> [!NOTE]
> Перед запуском проверьте `config.yaml` или отредактируйте `live_simulator.py`, чтобы путь к файлу пакетов (строка 50) соответствовал вашим данным.

---

## Альтернатива: Быстрый тест (без Nautilus)
Если вам нужно быстро проверить WR на многих монетах без симуляции исполнения ордеров, используйте:
```bash
python test_smc_ote_all_coins_fast.py
```
