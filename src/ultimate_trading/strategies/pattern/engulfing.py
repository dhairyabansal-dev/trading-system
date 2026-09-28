from __future__ import annotations

from ...core.models import MarketData, Signal, StrategyResult
from ...core.strategy import Strategy
from ...indicators.technical import bearish_engulfing, bullish_engulfing


class EngulfingPatternStrategy(Strategy):
    name = "pattern"

    def __init__(self, confidence_on_hit: float = 0.6):
        if not 0 <= confidence_on_hit <= 1:
            raise ValueError("confidence_on_hit must be in [0, 1]")
        self.confidence_on_hit = confidence_on_hit

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < 3:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "insufficient history"})

        if bool(bullish_engulfing(history).iloc[-1]):
            return StrategyResult(self.name, Signal.BUY, 0.6, self.confidence_on_hit, {"pattern": "bullish_engulfing"})
        if bool(bearish_engulfing(history).iloc[-1]):
            return StrategyResult(self.name, Signal.SELL, -0.6, self.confidence_on_hit, {"pattern": "bearish_engulfing"})
        return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"pattern": "none"})
