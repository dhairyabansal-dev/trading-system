import pandas as pd
from ...core.models import MarketData,Signal,StrategyResult
from ...core.strategy import Strategy
from ...indicators.momentum import rate_of_change
class SimpleMomentumStrategy(Strategy):
    name="simple_momentum"
    def __init__(self,lookback=20,threshold=.03): self.lookback,self.threshold=lookback,threshold
    def analyze(self,market_data:MarketData)->StrategyResult:
        if len(market_data.history)<=self.lookback: return StrategyResult(self.name,Signal.HOLD,0,0,{"reason":"warmup"})
        roc=rate_of_change(market_data.history.close,self.lookback).iloc[-1]
        if pd.isna(roc): return StrategyResult(self.name,Signal.HOLD,0,0,{"reason":"nan"})
        score=max(-1,min(1,roc/max(self.threshold,1e-9)))
        signal=Signal.BUY if roc>=self.threshold else Signal.SELL if roc<=-self.threshold else Signal.HOLD
        return StrategyResult(self.name,signal,score,min(1,abs(score)),{"roc":float(roc)})
