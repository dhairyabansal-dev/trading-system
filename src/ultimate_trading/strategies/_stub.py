from ..core.models import MarketData, Signal, StrategyResult
from ..core.strategy import Strategy
class StubStrategy(Strategy):
    def __init__(self,name:str): self.name=name
    def analyze(self,market_data:MarketData)->StrategyResult:
        return StrategyResult(self.name,Signal.HOLD,0.0,0.0,{"status":"stub"})
