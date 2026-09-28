"""
Simple momentum strategy — the Milestone 0 vertical slice.

Deliberately boring: a rate-of-change signal with a fixed threshold, and
confidence scaled down when the regime detector says we're RANGING (chop
is where naive momentum strategies bleed money). The point of this
strategy is to exercise the full pipeline end-to-end, not to demonstrate
edge — see docs/ARCHITECTURE.md Milestone 0 scope.
"""
from __future__ import annotations

import pandas as pd

from core.contracts import MarketData, Regime, Signal, StrategyResult
from core.indicators.momentum import rate_of_change
from core.regime.detector import detect_regime
from strategies.base import Strategy


class SimpleMomentumStrategy(Strategy):
    name = "simple_momentum"

    def __init__(
        self,
        lookback: int = 20,
        buy_threshold: float = 0.03,
        sell_threshold: float = -0.03,
        ranging_confidence_penalty: float = 0.5,
    ):
        self.lookback = lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.ranging_confidence_penalty = ranging_confidence_penalty

    def fit(self, train_data: pd.DataFrame) -> None:
        # No tunable parameters fit from data in this simple version.
        # Hook kept so the interface is exercised and so a future version
        # (e.g. threshold chosen from train-split volatility) has a home.
        return None

    def analyze(self, market_data: MarketData) -> StrategyResult:
        history = market_data.history
        if len(history) < self.lookback + 1:
            return StrategyResult(
                strategy_name=self.name,
                signal=Signal.HOLD,
                score=0.0,
                confidence=0.0,
                metadata={"reason": "insufficient history"},
            )

        roc = rate_of_change(history["close"], self.lookback).iloc[-1]
        if pd.isna(roc):
            return StrategyResult(
                strategy_name=self.name,
                signal=Signal.HOLD,
                score=0.0,
                confidence=0.0,
                metadata={"reason": "roc is nan"},
            )

        regime = detect_regime(history)

        if roc >= self.buy_threshold:
            signal = Signal.BUY
        elif roc <= self.sell_threshold:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        # Score: normalize roc against threshold scale, clamp to [-1, 1].
        scale = max(self.buy_threshold, abs(self.sell_threshold), 1e-9)
        score = max(-1.0, min(1.0, roc / scale / 3.0))

        confidence = min(1.0, abs(roc) / scale)
        if regime == Regime.RANGING:
            confidence *= self.ranging_confidence_penalty
        elif regime == Regime.UNKNOWN:
            confidence *= 0.5

        return StrategyResult(
            strategy_name=self.name,
            signal=signal,
            score=score,
            confidence=confidence,
            metadata={"roc": roc, "regime": regime.value, "lookback": self.lookback},
        )
