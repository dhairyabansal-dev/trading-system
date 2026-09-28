import pandas as pd
import pytest

from backtesting.engine import BacktestEngine, WalkForwardSplitter
from core.contracts import MarketData, Signal, StrategyResult
from core.data.synthetic import generate_ohlcv
from core.risk.manager import RiskConfig, RiskManager
from ensemble.combiner import (
    PassThroughCombiner,
    StrategyTrackRecord,
    WeightedVoteCombiner,
)
from strategies.momentum.simple_momentum import SimpleMomentumStrategy


def test_strategy_result_validates_score_and_confidence_bounds():
    with pytest.raises(ValueError):
        StrategyResult(strategy_name="x", signal=Signal.BUY, score=1.5, confidence=0.5)
    with pytest.raises(ValueError):
        StrategyResult(strategy_name="x", signal=Signal.BUY, score=0.5, confidence=1.5)


def test_synthetic_data_is_deterministic():
    a = generate_ohlcv(n_bars=200, seed=1)
    b = generate_ohlcv(n_bars=200, seed=1)
    pd.testing.assert_frame_equal(a, b)

    c = generate_ohlcv(n_bars=200, seed=2)
    assert not a["close"].equals(c["close"])


def test_momentum_strategy_holds_on_insufficient_history():
    strategy = SimpleMomentumStrategy(lookback=20)
    data = generate_ohlcv(n_bars=5, seed=1)
    md = MarketData(
        symbol="TEST",
        timestamp=data.index[-1],
        open=data.iloc[-1]["open"],
        high=data.iloc[-1]["high"],
        low=data.iloc[-1]["low"],
        close=data.iloc[-1]["close"],
        volume=data.iloc[-1]["volume"],
        history=data,
    )
    result = strategy.analyze(md)
    assert result.signal == Signal.HOLD
    assert result.confidence == 0.0


def test_momentum_strategy_produces_buy_on_strong_uptrend():
    strategy = SimpleMomentumStrategy(lookback=10, buy_threshold=0.03)
    # Strong deterministic uptrend, no noise: guarantees ROC exceeds threshold.
    data = generate_ohlcv(n_bars=60, seed=1, annualized_drift=5.0, annualized_vol=0.01)
    md = MarketData(
        symbol="TEST",
        timestamp=data.index[-1],
        open=data.iloc[-1]["open"],
        high=data.iloc[-1]["high"],
        low=data.iloc[-1]["low"],
        close=data.iloc[-1]["close"],
        volume=data.iloc[-1]["volume"],
        history=data,
    )
    result = strategy.analyze(md)
    assert result.signal == Signal.BUY
    assert result.score > 0


def test_passthrough_combiner_rejects_multiple_results():
    combiner = PassThroughCombiner()
    r = StrategyResult(strategy_name="a", signal=Signal.HOLD, score=0.0, confidence=0.0)
    with pytest.raises(ValueError):
        combiner.combine([r, r])


def test_weighted_vote_combiner_zero_weights_unvalidated_strategy():
    combiner = WeightedVoteCombiner(min_oos_trades=30)
    r = StrategyResult(strategy_name="unvalidated", signal=Signal.BUY, score=0.9, confidence=0.9)
    result = combiner.combine([r])
    assert result.signal == Signal.HOLD
    assert result.confidence == 0.0


def test_weighted_vote_combiner_uses_validated_strategy():
    combiner = WeightedVoteCombiner(min_oos_trades=30)
    combiner.register_track_record(
        StrategyTrackRecord(strategy_name="validated", out_of_sample_trades=90, out_of_sample_win_rate=0.55)
    )
    r = StrategyResult(strategy_name="validated", signal=Signal.BUY, score=0.9, confidence=0.9)
    result = combiner.combine([r])
    assert result.signal == Signal.BUY
    assert result.confidence > 0.0


def test_walk_forward_split_is_chronological_and_non_overlapping():
    data = generate_ohlcv(n_bars=100, seed=1)
    splitter = WalkForwardSplitter(train_fraction=0.7)
    train, test = splitter.split(data)
    assert len(train) == 70
    assert len(test) == 30
    assert train.index.max() < test.index.min()


def test_backtest_engine_runs_end_to_end_and_reports_both_splits():
    data = generate_ohlcv(n_bars=300, seed=1, regime_shift_at=150)
    engine = BacktestEngine(
        strategy=SimpleMomentumStrategy(lookback=20),
        combiner=PassThroughCombiner(),
        risk_manager=RiskManager(RiskConfig()),
        starting_cash=10_000.0,
    )
    report = engine.run(symbol="SYNTH/USDT", data=data)
    assert report.train.n_trades >= 0
    assert report.test.n_trades >= 0
    # sanity: metrics dict is serializable and complete
    assert set(report.train.as_dict().keys()) == {
        "total_return", "sharpe", "max_drawdown", "win_rate", "n_trades"
    }


def test_risk_manager_halts_on_drawdown():
    rm = RiskManager(RiskConfig(max_drawdown_halt=0.1))
    r = StrategyResult(strategy_name="x", signal=Signal.BUY, score=0.9, confidence=0.9)
    decision = rm.size(r, current_drawdown=0.15)
    assert decision.signal == Signal.HOLD
    assert decision.target_position_fraction == 0.0


def test_risk_manager_holds_below_confidence_floor():
    rm = RiskManager(RiskConfig(min_confidence_to_act=0.5))
    r = StrategyResult(strategy_name="x", signal=Signal.BUY, score=0.9, confidence=0.1)
    decision = rm.size(r, current_drawdown=0.0)
    assert decision.signal == Signal.HOLD
