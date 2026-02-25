# Актуальная документация по Nautilus Trader (версия 1.122.0+)

Ниже приведены основные критичные изменения, обнаруженные при переходе на новые версии Nautilus Trader (1.122.0 и выше), а также особенности работы с Mock-инфраструктурой и балансами для **Спотового (CASH)** аккаунта.

## 1. Изменение работы с балансами аккаунта
Ранее доступ к балансу (например, USDT, BTC) осуществлялся через метод `.balance()` или `balances_c()`. 
**Теперь эти методы устарели (Deprecated / Broken).**

**Новый метод доступа:**
```python
# Получить весь баланс по валюте:
account.balance_total(USDT) 

# Получить заблокированный (в ордерах) баланс:
account.balances_locked(USDT)

# Получить свободный баланс:
account.balances_free(USDT)
```

## 2. Реализация Mock Execution Client
Если вы пишете свой кастомный `ExecutionClient` или `MockExecutionClient` (унаследованный от `LiveExecutionClient`), есть строгие правила работы с Cython-классами движка:

### А. Идентификатор аккаунта (`account_id`)
Запрещено напрямую перезаписывать `self.account_id`, так как в базовом Cython классе это свойство (Property) только для чтения, либо инициализируется иначе.
Правильный подход — вызвать внутренний метод `_set_account_id` в конструкторе:
```python
def __init__(self, ...):
    super().__init__(...)
    self.mock_account_id = AccountId("BINANCE-SPOT")
    self._set_account_id(self.mock_account_id)
```

### Б. Отправка состояния аккаунта (`generate_account_state`)
Ранее можно было собрать объект `AccountState` самостоятельно (с указанием `account_type=AccountType.CASH` для спота) и передать его.
**Сейчас сигнатура функции `generate_account_state` изменилась.** Она принимает распакованные параметры напрямую:
```python
self.generate_account_state(
    balances=[
        AccountBalance(total=Money(1000, USDT), locked=Money(0, USDT), free=Money(1000, USDT)),
        AccountBalance(total=Money(1, BTC), locked=Money(0, BTC), free=Money(1, BTC))
    ],
    margins=[],  # Пусто для спотового аккаунта
    reported=True,
    ts_event=clock.timestamp_ns(),
    info={}
)
```
*Примечание:* Движок Nautilus Trader сам производит вычет балансов (маржу и стоимость маркировки), когда вы отправляете ордера на исполнение (`generate_order_filled`). Попытка вручную изменять состояние аккаунта через `generate_account_state` повторно после сделки на моно-валютных или спотовых аккаунтах может привести к ошибкам дублирования транзакций (`ValueError: single-currency account has multiple currency update`).

### В. Отправка исполнения ордеров (`generate_order_filled`)
Обязательные параметры сильно типизированы. Особое внимание стоит уделить `LiquiditySide` и `Price`.
```python
from nautilus_trader.model.enums import LiquiditySide

self.generate_order_filled(
    strategy_id=order.strategy_id,
    instrument_id=order.instrument_id,
    client_order_id=order.client_order_id,
    venue_order_id=VenueOrderId(f"mock_{order.client_order_id}"),
    venue_position_id=None,
    trade_id=TradeId("unique-string-uuid"),
    order_side=order.side,
    order_type=order.order_type,
    last_qty=order.quantity,
    last_px=price,  # Обязательно объект Price! Не Decimal.
    quote_currency=USDT,
    commission=Money(0, USDT), # Комиссии теперь передаются объектом Money
    liquidity_side=LiquiditySide.TAKER if order.order_type == OrderType.MARKET else LiquiditySide.MAKER, 
    ts_event=clock.timestamp_ns(),
)
```

## 3. Особенности работы со Спотовым рынком (Spot Account)
Для спота:
1. `AccountType` должен быть `AccountType.CASH`. В стандартных Execution клиентах (например, Binance) это задается в конфигурации клиента (`BinanceSpotExecutionClient`).
2. В споте нет `margin` и `leverage`.
3. При торговле база (Base Currency) списывается напрямую в обмен на квоты (Quote Currency), никаких синтетических PnL перерасчетов, как на фьючерсах. При создании Custom Mock клиента `AccountBalance` должен явно менять `free` и `locked`, если вы пишете свой эмулятор стакана, или, в случае с прокидыванием в Live Node, движок сделает это сам (Portfolio manager).

## 4. Ошибки типизации (TypeError и AttributeError)
- Сложение/вычитание `Decimal` и `Price` (например `tick.price - 1000`) приведет к падению, так как `tick.price` это `Price`.
  Решение: `Price(tick.price.as_double() - 1000.0, tick.price.precision)`.
- Идентификаторы: `ClientOrderId`, `TradeId`, `VenueOrderId` — это всегда типизированные классы, оборачивающие строки: `TradeId(str(uuid4()))`.

## 5. Live/Standalone Mock режим без DataClient
Если вы используете `TradingNode` с MockExecutionClient, но без реального DataClient (например, для юнит-тестов исполнения алгоритма):
1. Не вызывайте `self.subscribe_trade_ticks(self.instrument_id)` в `on_start()`, так как нет Дата-движка для подписки и произойдет ошибка.
2. Пробрасывайте тики вручную напрямую в `strategy.on_trade_tick(tick)` и в ядро `node.kernel.data_engine._handle_data(tick)`, чтобы обновить внутренние кэши движка (`self._cache.quote_tick(...)`).
