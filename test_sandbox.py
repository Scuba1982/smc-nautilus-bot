"""
test_sandbox.py — Проверка совместимости NautilusTrader API.
НЕ трогает основной код. Запускать: python test_sandbox.py
"""
import inspect
import sys

OK = 0
FAIL = 0

def check(label, fn):
    global OK, FAIL
    try:
        result = fn()
        print(f"  OK  {label}" + (f" -> {result}" if result is not None else ""))
        OK += 1
        return result
    except Exception as e:
        print(f"  ERR {label}: {e}")
        FAIL += 1
        return None


print("=" * 60)
print("Тест 1: Импорты")
print("=" * 60)

LiveExecClientFactory = check(
    "from nautilus_trader.live.factories import LiveExecClientFactory",
    lambda: __import__('nautilus_trader.live.factories', fromlist=['LiveExecClientFactory']).LiveExecClientFactory
)

check(
    "ExecClientFactory (старое имя — должно УПАСТЬ)",
    lambda: __import__('nautilus_trader.live.factories', fromlist=['ExecClientFactory']).ExecClientFactory
)

SandboxExecutionClient = check(
    "SandboxExecutionClient",
    lambda: __import__('nautilus_trader.adapters.sandbox.execution', fromlist=['SandboxExecutionClient']).SandboxExecutionClient
)

SandboxExecutionClientConfig = check(
    "SandboxExecutionClientConfig",
    lambda: __import__('nautilus_trader.adapters.sandbox.execution', fromlist=['SandboxExecutionClientConfig']).SandboxExecutionClientConfig
)

LiveExecEngineConfig = check(
    "LiveExecEngineConfig",
    lambda: __import__('nautilus_trader.config', fromlist=['LiveExecEngineConfig']).LiveExecEngineConfig
)

print()
print("=" * 60)
print("Тест 2: Сигнатуры методов")
print("=" * 60)

if LiveExecClientFactory:
    check(
        "LiveExecClientFactory.create signature",
        lambda: str(inspect.signature(LiveExecClientFactory.create))
    )

if SandboxExecutionClient:
    check(
        "SandboxExecutionClient.__init__ signature",
        lambda: str(inspect.signature(SandboxExecutionClient.__init__))
    )

print()
print("=" * 60)
print("Тест 3: TradingNodeConfig + reconciliation=False")
print("=" * 60)

def test_config():
    from nautilus_trader.config import TradingNodeConfig, LoggingConfig
    from nautilus_trader.config import LiveExecEngineConfig
    cfg = TradingNodeConfig(
        trader_id="TEST-001",
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=LiveExecEngineConfig(reconciliation=False),
        data_clients={},
        exec_clients={},
    )
    return f"reconciliation={cfg.exec_engine.reconciliation}"

check("TradingNodeConfig(exec_engine=LiveExecEngineConfig(reconciliation=False))", test_config)

print()
print("=" * 60)
print("Тест 4: Правильная фабрика для Sandbox")
print("=" * 60)

def test_factory_subclass():
    from nautilus_trader.live.factories import LiveExecClientFactory
    from nautilus_trader.adapters.sandbox.execution import SandboxExecutionClient, SandboxExecutionClientConfig

    class TestSandboxFactory(LiveExecClientFactory):
        @staticmethod
        def create(loop, name, config, msgbus, cache, clock):
            # Проверяем — portfolio нет в сигнатуре
            return f"factory create called: name={name}"

    sig = str(inspect.signature(TestSandboxFactory.create))
    return f"subclass OK, create: {sig}"

check("class TestSandboxFactory(LiveExecClientFactory)", test_factory_subclass)

print()
print("=" * 60)
print(f"ИТОГ: {OK} OK, {FAIL} FAIL")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
