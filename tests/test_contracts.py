import pytest
from ultimate_trading.core.models import Signal,StrategyResult
def test_bounds():
    StrategyResult("x",Signal.BUY,1,1)
    with pytest.raises(ValueError):StrategyResult("x",Signal.BUY,2,1)
