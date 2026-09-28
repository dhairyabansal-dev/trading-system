from __future__ import annotations

import pandas as pd

from ..core.models import Regime
from ..indicators.momentum import rolling_volatility, sma


class RegimeDetector:
    """Deterministic baseline regime classifier.

    It uses trend separation (fast/slow SMA) plus volatility relative to its
    recent baseline. It deliberately returns UNKNOWN during warm-up instead
    of inventing a regime.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        volatility_window: int = 20,
        high_volatility_ratio: float = 1.8,
        low_volatility_ratio: float = 0.65,
    ):
        if not 1 <= fast_window < slow_window:
            raise ValueError("require 1 <= fast_window < slow_window")
        if volatility_window < 2:
            raise ValueError("volatility_window must be >= 2")
        if high_volatility_ratio <= 1:
            raise ValueError("high_volatility_ratio must be > 1")
        if not 0 < low_volatility_ratio < 1:
            raise ValueError("low_volatility_ratio must be in (0, 1)")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.volatility_window = volatility_window
        self.high_volatility_ratio = high_volatility_ratio
        self.low_volatility_ratio = low_volatility_ratio

    def detect(self, history: pd.DataFrame) -> Regime:
        required = self.slow_window + self.volatility_window
        if len(history) < required:
            return Regime.UNKNOWN

        close = history["close"]
        fast = sma(close, self.fast_window).iloc[-1]
        slow = sma(close, self.slow_window).iloc[-1]
        volatility = rolling_volatility(close, self.volatility_window)
        recent = volatility.iloc[-1]
        baseline = volatility.iloc[-(self.volatility_window * 3):].mean()

        if any(pd.isna(v) for v in (fast, slow, recent, baseline)) or baseline <= 0:
            return Regime.UNKNOWN

        ratio = recent / baseline
        if ratio >= self.high_volatility_ratio:
            return Regime.HIGH_VOLATILITY
        if ratio <= self.low_volatility_ratio:
            return Regime.LOW_VOLATILITY

        separation = abs(fast - slow) / max(abs(slow), 1e-12)
        if separation < 0.0025:
            return Regime.RANGING
        return Regime.TRENDING_UP if fast > slow else Regime.TRENDING_DOWN
