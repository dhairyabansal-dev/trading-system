from __future__ import annotations

import pandas as pd

from ...core.models import MarketData, Signal, StrategyResult
from ...core.strategy import Strategy
from ...indicators.momentum import sma
from ...indicators.technical import linreg_slope


class SMATrendStrategy(Strategy):
    name = "trend"

    def __init__(self, fast_window: int = 10, slow_window: int = 30, slope_window: int = 20):
        if not 1 <= fast_window < slow_window:
            raise ValueError("require 1 <= fast_window < slow_window")
        if slope_window < 2:
            raise ValueError("slope_window must be >= 2")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.slope_window = slope_window

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < max(self.slow_window, self.slope_window) + 2:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "insufficient history"})

        close = history["close"]
        fast = sma(close, self.fast_window).iloc[-1]
        slow = sma(close, self.slow_window).iloc[-1]
        slope = linreg_slope(close, self.slope_window).iloc[-1]

        if any(pd.isna(v) for v in (fast, slow, slope)) or slow == 0:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "warming up"})

        cross_strength = (fast - slow) / slow
        signal = Signal.BUY if fast > slow else Signal.SELL if fast < slow else Signal.HOLD
        score = max(-1.0, min(1.0, (cross_strength * 20 + slope * 50) / 2))
        confidence = min(1.0, abs(slope) * 30)

        return StrategyResult(
            self.name, signal, score, confidence,
            {"fast_sma": fast, "slow_sma": slow, "slope": slope},
        )
