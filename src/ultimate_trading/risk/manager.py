from dataclasses import dataclass
from ..core.models import RiskAdjustedSignal,Signal,StrategyResult
@dataclass(frozen=True)
class RiskConfig:
    max_position_fraction:float=.25
    min_confidence:float=.2
    max_drawdown_halt:float=.25
class RiskManager:
    def __init__(self,config=None): self.config=config or RiskConfig()
    def apply(self,result:StrategyResult,drawdown:float)->RiskAdjustedSignal:
        if drawdown>=self.config.max_drawdown_halt or result.confidence<self.config.min_confidence:return RiskAdjustedSignal(Signal.HOLD,0.0,"risk gate")
        fraction=min(abs(result.score)*result.confidence,self.config.max_position_fraction)
        return RiskAdjustedSignal(result.signal,fraction if result.signal!=Signal.HOLD else 0.0,"bounded risk allocation")
