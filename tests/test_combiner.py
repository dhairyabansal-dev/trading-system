from ultimate_trading.core.models import Signal, StrategyResult
from ultimate_trading.ensemble.combiner import StrategyTrackRecord, WeightedVoteCombiner


def test_combiner_ignores_unvalidated_strategies():
    combiner = WeightedVoteCombiner(
        records={
            "good": StrategyTrackRecord("good", oos_trades=50, validated=True, oos_sharpe=1.0),
            "bad": StrategyTrackRecord("bad", oos_trades=50, validated=False, oos_sharpe=5.0),
        }
    )
    result = combiner.combine([
        StrategyResult("good", Signal.BUY, 0.8, 0.9),
        StrategyResult("bad", Signal.SELL, -1.0, 1.0),
    ])
    assert result.signal == Signal.BUY
    assert result.metadata["contributors"] == ["good"]


def test_combiner_holds_without_validation_evidence():
    combiner = WeightedVoteCombiner(records={})
    result = combiner.combine([StrategyResult("x", Signal.BUY, 1.0, 1.0)])
    assert result.signal == Signal.HOLD
    assert result.confidence == 0
