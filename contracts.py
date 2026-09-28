"""
Core data contracts. Every strategy, the ensemble, risk, and portfolio
layers speak these types and nothing else. Do not special-case a
particular strategy family anywhere downstream of Strategy.analyze().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MarketData:
    """One point-in-time view of a symbol, with rolling history attached.

    `history` must contain only bars up to and including `timestamp` —
    callers (the backtester) are responsible for never leaking future bars
    into this object. Strategies should treat `history` as read-only.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    history: pd.DataFrame  # columns: open, high, low, close, volume; indexed by timestamp


@dataclass(frozen=True)
class StrategyResult:
    strategy_name: str
    signal: Signal
    score: float        # -1.0 (max bearish) .. 1.0 (max bullish)
    confidence: float   # 0.0 (ignore me) .. 1.0 (trust me fully)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not (-1.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [-1, 1], got {self.score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class RiskAdjustedSignal:
    """Output of the risk layer: a StrategyResult plus a concrete, bounded
    position sizing decision. This is still paper-only — nothing here talks
    to a broker or exchange."""
    signal: Signal
    target_position_fraction: float  # fraction of portfolio equity, 0..max_position_fraction
    reason: str
