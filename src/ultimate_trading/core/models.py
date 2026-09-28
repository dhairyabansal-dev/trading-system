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
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    history: pd.DataFrame

@dataclass(frozen=True)
class StrategyResult:
    strategy_name: str
    signal: Signal
    score: float
    confidence: float
    metadata: dict = field(default_factory=dict)
    def __post_init__(self):
        if not -1 <= self.score <= 1: raise ValueError("score must be in [-1, 1]")
        if not 0 <= self.confidence <= 1: raise ValueError("confidence must be in [0, 1]")

@dataclass(frozen=True)
class RiskAdjustedSignal:
    signal: Signal
    target_position_fraction: float
    reason: str
