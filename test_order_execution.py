import asyncio
from decimal import Decimal

from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import MessageBus, TestClock
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, OrderType, AggressorSide, TriggerType, LiquiditySide
from nautilus_trader.model.identifiers import Venue, InstrumentId, AccountId, ClientOrderId, VenueOrderId, TraderId, ClientId, Symbol, TradeId
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.live.node import TradingNode
from nautilus_trader.live.execution_client import LiveExecutionClient
from nautilus_trader.execution.messages import SubmitOrder, CancelOrder, ModifyOrder, CancelAllOrders, BatchCancelOrders
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.config import LiveExecEngineConfig
from nautilus_trader.accounting.accounts.margin import MarginAccount
from nautilus_trader.model.events.account import AccountState
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.objects import AccountBalance

class MockExecutionClient(LiveExecutionClient):
    def __init__(self, msgbus, cache, clock):
        super().__init__(
            loop=asyncio.get_running_loop(),
            client_id=ClientId("MOCK"),
            venue=Venue("MOCK"),
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            instrument_provider=InstrumentProvider(),
            msgbus=msgbus,
            cache=cache,
            clock=clock,
        )
        self.trigger_orders = [] # Для стопов
        self.limit_orders = []   # Для лимиток
        
        self.usdt_balance = 100000.0
        self.btc_balance = 0.0
        self.mock_account_id = AccountId("MOCK-001")
        self._set_account_id(self.mock_account_id)

    async def _connect(self):
        self._set_connected(True)
        # Отправляем начальный баланс
        self.update_account_state()

    async def _disconnect(self):
        self._set_connected(False)

    def update_account_state(self):
        self.generate_account_state(
            balances=[
                AccountBalance(total=Money(self.usdt_balance, USDT), locked=Money(0, USDT), free=Money(self.usdt_balance, USDT)),
                AccountBalance(total=Money(self.btc_balance, BTC), locked=Money(0, BTC), free=Money(self.btc_balance, BTC))
            ],
            margins=[],
            reported=True,
            ts_event=self._clock.timestamp_ns(),
            info={"account_type": AccountType.CASH}
        )

    async def _submit_order(self, command: SubmitOrder):
        order = command.order
        print(f"\n[Биржа] Получен ордер: {order.client_order_id} ({order.order_type}, {order.side})")
        
        self.generate_order_accepted(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=VenueOrderId(f"mock_{order.client_order_id}"),
            ts_event=self._clock.timestamp_ns()
        )

        fill_price = None
        if order.order_type == OrderType.MARKET:
            quote = self._cache.quote_tick(order.instrument_id)
            if quote:
                fill_price = quote.ask if order.side == OrderSide.BUY else quote.bid
            else:
                last_trade = self._cache.trade_tick(order.instrument_id)
                if last_trade:
                    fill_price = last_trade.price
                    
            if fill_price:
                self._fill_order(order, fill_price)
            else:
                print(f"[Биржа] Не могу исполнить маркет. Нет цен. Выполним по 60000.00 для теста.")
                self._fill_order(order, Price.from_str("60000.00"))
                
        elif order.order_type == OrderType.LIMIT:
            self.limit_orders.append(order)
            print(f"[Биржа] Лимитный ордер установлен на {order.price}")
            
        elif order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            self.trigger_orders.append(order)
            print(f"[Биржа] Стоп-ордер установлен с триггером {order.trigger_price}")

    def _fill_order(self, order, price):
        qty_double = order.quantity.as_double()
        price_double = price.as_double()
        
        cost = qty_double * price_double
        
        if order.side == OrderSide.BUY:
            self.usdt_balance -= cost
            self.btc_balance += qty_double
        else:
            self.usdt_balance += cost
            self.btc_balance -= qty_double
            
        self.generate_order_filled(
            strategy_id=order.strategy_id,
            instrument_id=order.instrument_id,
            client_order_id=order.client_order_id,
            venue_order_id=VenueOrderId(f"mock_{order.client_order_id}"),
            venue_position_id=None,
            trade_id=TradeId(UUID4().value),
            order_side=order.side,
            order_type=order.order_type,
            last_qty=order.quantity,
            last_px=price,
            quote_currency=USDT,
            commission=Money(0, USDT),
            liquidity_side=LiquiditySide.TAKER if order.order_type == OrderType.MARKET else LiquiditySide.MAKER, 
            ts_event=self._clock.timestamp_ns(),
        )
        print(f"[Биржа] Ордер {order.client_order_id} ИСПОЛНЕН по {price}. Engine спишет баланс.")
        
    def check_triggers_and_limits(self, current_price: Price):
        current_price_double = current_price.as_double()
        
        for order in list(self.trigger_orders):
            triggered = False
            trigger = order.trigger_price.as_double()
            if order.side == OrderSide.SELL and current_price_double <= trigger:
                triggered = True
            elif order.side == OrderSide.BUY and current_price_double >= trigger:
                triggered = True
                
            if triggered:
                print(f"\n[Биржа] ТРИГГЕР СРАБОТАЛ для {order.client_order_id}!")
                self.trigger_orders.remove(order)
                self._fill_order(order, current_price)

        for order in list(self.limit_orders):
            filled = False
            limit = order.price.as_double()
            if order.side == OrderSide.SELL and current_price_double >= limit:
                filled = True
            elif order.side == OrderSide.BUY and current_price_double <= limit:
                filled = True
                
            if filled:
                print(f"\n[Биржа] ЛИМИТКА ИСПОЛНЕНА для {order.client_order_id}!")
                self.limit_orders.remove(order)
                self._fill_order(order, order.price)

    async def _cancel_order(self, command: CancelOrder):
        pass
    async def _modify_order(self, command: ModifyOrder):
        pass
    async def _cancel_all_orders(self, command: CancelAllOrders):
        pass
    async def _submit_order_list(self, command):
        pass
    async def _batch_cancel_orders(self, command: BatchCancelOrders):
        pass
    async def generate_order_status_reports(self, instrument_id=None, start=None, end=None):
        return []
    async def generate_position_status_reports(self, instrument_id=None):
        return []

class SimpleTestStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str

class SimpleTestStrategy(Strategy):
    def __init__(self, config: SimpleTestStrategyConfig) -> None:
        super().__init__(config)
        self.inst_id = InstrumentId.from_str(config.instrument_id)
        self.entered = False

    def on_start(self) -> None:
        pass

    def on_trade_tick(self, tick: TradeTick) -> None:
        if not self.entered:
            self.log.info("--- СТРАТЕГИЯ: Отправляем Market Buy, TP (Limit Sell) и SL (Stop Sell) ---")
            qty = Quantity.from_str("1.0")
            
            # Покупка рынком
            buy_order = self.order_factory.market(
                instrument_id=self.inst_id,
                order_side=OrderSide.BUY,
                quantity=qty,
            )
            self.submit_order(buy_order)
            
            # Стоп-лосс на 1000 ниже
            stop_price = Price(tick.price.as_double() - 1000.0, tick.price.precision)
            stop_order = self.order_factory.stop_market(
                instrument_id=self.inst_id,
                order_side=OrderSide.SELL,
                quantity=qty,
                trigger_price=stop_price,
                trigger_type=TriggerType.LAST_PRICE,
            )
            self.submit_order(stop_order)
            
            # Тейк-профит на 1000 выше
            tp_price = Price(tick.price.as_double() + 1000.0, tick.price.precision)
            tp_order = self.order_factory.limit(
                instrument_id=self.inst_id,
                order_side=OrderSide.SELL,
                quantity=qty,
                price=tp_price,
            )
            self.submit_order(tp_order)
            
            self.entered = True

async def run_test():
    print("==========================================================")
    print(" Запуск тестирования ордеров в Live Mode с Mock Клиентом ")
    print("==========================================================")
    
    node_config = TradingNodeConfig(
        exec_engine=LiveExecEngineConfig(
            reconciliation=False,
        )
    )
    node = TradingNode(config=node_config)
    clock = node.kernel.clock
    msgbus = node.kernel.msgbus
    cache = node.kernel.cache
    
    inst_id = InstrumentId.from_str("BTCUSDT.MOCK")
    instrument = CurrencyPair(
        instrument_id=inst_id,
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        price_precision=2,
        size_precision=4,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.0001"),
        lot_size=Quantity.from_str("0.0001"),
        max_quantity=Quantity.from_str("100.0"),
        min_quantity=Quantity.from_str("0.0001"),
        max_notional=None,
        min_notional=None,
        max_price=Price.from_str("1000000.0"),
        min_price=Price.from_str("0.01"),
        margin_init=Decimal("1.0"),
        margin_maint=Decimal("1.0"),
        maker_fee=Decimal("0.0"),
        taker_fee=Decimal("0.0"),
        ts_event=clock.timestamp_ns(),
        ts_init=clock.timestamp_ns()
    )
    cache.add_instrument(instrument)
    
    mock_client = MockExecutionClient(msgbus=msgbus, cache=cache, clock=clock)
    node.kernel.exec_engine.register_client(mock_client)
    
    config = SimpleTestStrategyConfig(instrument_id=str(inst_id))
    strategy = SimpleTestStrategy(config=config)
    node.trader.add_strategy(strategy)
    
    node.build()
    node_task = asyncio.create_task(node.run_async())
    await asyncio.sleep(1)
    
    account_id = mock_client.mock_account_id

    # Начальный баланс
    print("\n[Проверка] Начальный баланс:")
    acc = cache.account(account_id)
    if acc:
        print("  USDT:", acc.balance_total(USDT))
        print("   BTC:", acc.balance_total(BTC))

    # Первый тик
    print("\n[Данные] Отправляем цену 60000.00 -> сработает логика входа стратегии")
    tick1 = TradeTick(
        instrument_id=inst_id, price=Price.from_str("60000.00"), size=Quantity.from_str("1.0"),
        aggressor_side=AggressorSide.BUYER, trade_id=TradeId("1"), ts_event=clock.timestamp_ns(), ts_init=clock.timestamp_ns()
    )
    node.kernel.data_engine._handle_data(tick1)
    strategy.on_trade_tick(tick1)
    mock_client.check_triggers_and_limits(tick1.price) # если были ордера
    
    await asyncio.sleep(0.5)
    
    # Баланс после рыночной покупки (1 BTC за 60000 USDT)
    print("\n[Проверка] Баланс после Market покупки:")
    if acc:
        print("  USDT:", acc.balance_total(USDT), "(Ожидаем 40000.0)")
        print("   BTC:", acc.balance_total(BTC), "(Ожидаем 1.0)")

    # Второй тик - выбивает тейк профит (61000)
    print("\n[Данные] Отправляем цену 61000.00 -> сработает лимитный Take Profit")
    tick2 = TradeTick(
        instrument_id=inst_id, price=Price.from_str("61000.00"), size=Quantity.from_str("1.0"),
        aggressor_side=AggressorSide.BUYER, trade_id=TradeId("2"), ts_event=clock.timestamp_ns(), ts_init=clock.timestamp_ns()
    )
    node.kernel.data_engine._handle_data(tick2)
    strategy.on_trade_tick(tick2)
    mock_client.check_triggers_and_limits(tick2.price)
    
    await asyncio.sleep(0.5)
    
    # Баланс после тейка (продали 1 BTC за 61000 USDT)
    print("\n[Проверка] Баланс после Take Profit (Limit Sell):")
    if acc:
        print("  USDT:", acc.balance_total(USDT), "(Ожидаем 101000.0)")
        print("   BTC:", acc.balance_total(BTC), "(Ожидаем 0.0)")

    node.stop()

if __name__ == "__main__":
    asyncio.run(run_test())
