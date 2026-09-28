from __future__ import annotations

import pandas as pd

from ...core.models import MarketData, Signal, StrategyResult
from ...core.strategy import Strategy
from ...indicators.technical import volume_delta


class VolumeDeltaProxyStrategy(Strategy):
    name = "orderflow"

    def __init__(self, window: int = 20, z_entry: float = 1.5):
        if window < 2 or z_entry <= 0:
            raise ValueError("window must be >= 2 and z_entry must be positive")
        self.window = window
        self.z_entry = z_entry

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < self.window * 3:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "insufficient history"})

        delta = volume_delta(history, self.window)
        recent = delta.iloc[-1]
        std = delta.rolling(self.window * 2, min_periods=self.window).std().iloc[-1]
        if pd.isna(recent) or pd.isna(std) or std <= 0:
            return StrategyResult(self.name, Signal.HOLD, 0.0, 0.0, {"reason": "warming up"})

        z = recent / std
        if z >= self.z_entry:
            signal, score = Signal.BUY, min(1.0, z / (2 * self.z_entry))
        elif z <= -self.z_entry:
            signal, score = Signal.SELL, -min(1.0, abs(z) / (2 * self.z_entry))
        else:
            signal, score = Signal.HOLD, 0.0

        confidence = min(0.6, abs(z) / (2 * self.z_entry)) if signal != Signal.HOLD else 0.0
        return StrategyResult(
            self.name, signal, score, confidence,
            {"volume_delta_z": z, "is_proxy": True},
        )
