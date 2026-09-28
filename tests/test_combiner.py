from ultimate_trading.core.models import Signal,StrategyResult
from ultimate_trading.ensemble.combiner import PassThroughCombiner,WeightedVoteCombiner
def test_passthrough():
    r=StrategyResult("s",Signal.BUY,.5,.8);assert PassThroughCombiner().combine([r])==r
def test_unvalidated_is_ignored():
    r=StrategyResult("s",Signal.BUY,1,1);out=WeightedVoteCombiner().combine([r]);assert out.signal==Signal.HOLD
