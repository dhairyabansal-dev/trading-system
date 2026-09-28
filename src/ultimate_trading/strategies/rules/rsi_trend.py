from __future__ import annotations

import pandas as pd

from ...core.models import MarketData, Signal, StrategyResult
from ...core.strategy import Strategy
from ...indicators.momentum import sma
from ...indicators.technical import rsi


class RSITrendRuleStrategy(Strategy):
    name = "rule_based"

    def __init__(
        self,
        rsi_window: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        trend_window: int = 50,
    ):
        if rsi_window < 2 or trend_window < 2:
            raise ValueError("windows must be >= 2")
        if not 0 < rsi_oversold < rsi_overbought < 100:
            raise ValueError("require 0 < oversold < overbought < 100")
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_window = trend_window

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < max(self.rsi_window, self.trend_window) + 2:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "insufficient history"})

        close = history["close"]
        value = rsi(close, self.rsi_window).iloc[-1]
        trend = sma(close, self.trend_window).iloc[-1]
        price = close.iloc[-1]

        if pd.isna(value) or pd.isna(trend):
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "warming up"})

        if value <= self.rsi_oversold and price > trend:
            score = min(1.0, (self.rsi_oversold - value) / self.rsi_oversold)
            return StrategyResult(self.name, Signal.BUY, score, 0.7, {"rsi": value, "trend_sma": trend, "rule": "buy_the_dip_above_trend"})
        if value >= self.rsi_overbought and price < trend:
            score = -min(1.0, (value - self.rsi_overbought) / (100 - self.rsi_overbought))
            return StrategyResult(self.name, Signal.SELL, score, 0.7, {"rsi": value, "trend_sma": trend, "rule": "sell_the_rip_below_trend"})
        return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"rsi": value, "trend_sma": trend, "rule": "no_rule_fired"})
