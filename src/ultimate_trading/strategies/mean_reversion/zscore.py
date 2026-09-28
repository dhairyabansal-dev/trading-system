from __future__ import annotations

import pandas as pd

from ...core.models import MarketData, Regime, Signal, StrategyResult
from ...core.strategy import Strategy
from ...indicators.technical import zscore
from ...regime.detector import RegimeDetector


class ZScoreMeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 2.0,
        trend_confidence_penalty: float = 0.3,
        regime_detector: RegimeDetector | None = None,
    ):
        if window < 2 or entry_z <= 0:
            raise ValueError("window must be >= 2 and entry_z must be positive")
        self.window = window
        self.entry_z = entry_z
        self.trend_confidence_penalty = trend_confidence_penalty
        self.regime_detector = regime_detector or RegimeDetector()

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < self.window + 5:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "insufficient history"})

        value = zscore(history["close"], self.window).iloc[-1]
        if pd.isna(value):
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "warming up"})

        regime = self.regime_detector.detect(history)
        if value <= -self.entry_z:
            signal = Signal.BUY
            score = min(1.0, abs(value) / (2 * self.entry_z))
        elif value >= self.entry_z:
            signal = Signal.SELL
            score = -min(1.0, abs(value) / (2 * self.entry_z))
        else:
            signal = Signal.HOLD
            score = 0.0

        confidence = min(1.0, abs(value) / self.entry_z) if signal != Signal.HOLD else 0.0
        if regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            confidence *= self.trend_confidence_penalty

        return StrategyResult(
            self.name,
            signal,
            score,
            confidence,
            {"zscore": value, "regime": regime.value},
        )
