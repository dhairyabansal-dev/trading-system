from __future__ import annotations

import pandas as pd

from ...core.models import MarketData, Signal, StrategyResult
from ...core.strategy import Strategy
from ...indicators.technical import atr, donchian_channel


class DonchianBreakoutStrategy(Strategy):
    name = "breakout"

    def __init__(self, window: int = 20, atr_window: int = 14):
        if window < 2 or atr_window < 2:
            raise ValueError("windows must be >= 2")
        self.window = window
        self.atr_window = atr_window

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < max(self.window, self.atr_window) + 2:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "insufficient history"})

        upper, lower = donchian_channel(history, self.window)
        current_atr = atr(history, self.atr_window).iloc[-1]
        previous_upper = upper.iloc[-2]
        previous_lower = lower.iloc[-2]
        close = float(history["close"].iloc[-1])

        if any(pd.isna(v) for v in (previous_upper, previous_lower, current_atr)) or current_atr <= 0:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "warming up"})

        if close > previous_upper:
            distance = (close - previous_upper) / current_atr
            score = min(1.0, distance / 2.0)
            signal = Signal.BUY
        elif close < previous_lower:
            distance = (previous_lower - close) / current_atr
            score = -min(1.0, distance / 2.0)
            signal = Signal.SELL
        else:
            score = 0.0
            signal = Signal.HOLD

        return StrategyResult(
            self.name,
            signal,
            score,
            min(1.0, abs(score)),
            {"donchian_upper": previous_upper, "donchian_lower": previous_lower, "atr": current_atr},
        )
