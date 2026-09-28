import pandas as pd
import pytest

from ultimate_trading.backtesting.engine import BacktestConfig, BacktestEngine, WalkForwardSplitter
from ultimate_trading.core.models import Signal, StrategyResult, MarketData
from ultimate_trading.core.strategy import Strategy
from ultimate_trading.data.loader import validate_ohlcv
from ultimate_trading.data.synthetic import generate_ohlcv
from ultimate_trading.strategies.momentum.simple import SimpleMomentumStrategy


def test_split():
    data = generate_ohlcv(100)
    s = WalkForwardSplitter(.7).split(data)
    assert len(s.train) == 70 and len(s.test) == 30
    assert s.train.index[-1] < s.test.index[0]


def test_ohlcv_validation_rejects_naive_index():
    data = generate_ohlcv(10)
    data.index = data.index.tz_localize(None)
    with pytest.raises(ValueError):
        validate_ohlcv(data)


def test_momentum_contract():
    data = generate_ohlcv(100)
    row = data.iloc[-1]
    result = SimpleMomentumStrategy(lookback=10, threshold=.01).analyze(
        MarketData("TEST", data.index[-1].to_pydatetime(), *map(float, row), data)
    )
    assert result.signal in Signal
    assert -1 <= result.score <= 1
    assert 0 <= result.confidence <= 1


class AlwaysLong(Strategy):
    name = "always_long"

    def analyze(self, market_data: MarketData) -> StrategyResult:
        return StrategyResult(self.name, Signal.BUY, 1.0, 1.0)


def test_backtest_produces_ledger_and_metrics():
    data = generate_ohlcv(250, seed=7)
    result = BacktestEngine(
        BacktestConfig(initial_capital=100_000, fee_bps=0, slippage_bps=0)
    ).run(data, AlwaysLong())
    assert result.trade_count == 1
    assert len(result.equity_curve) > 0
    assert result.equity_curve.iloc[-1] > 0
    assert result.max_drawdown >= 0
