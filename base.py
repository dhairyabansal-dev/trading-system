"""
The one interface every strategy family (grid, DCA, momentum, pattern,
order-flow, rule-based, etc.) must implement. Nothing downstream of
`analyze()` is allowed to special-case a particular strategy — that
constraint is what lets 20 strategies plug into the same combiner without
rewriting the system.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.contracts import MarketData, StrategyResult


class Strategy(ABC):
    name: str = "unnamed_strategy"

    @abstractmethod
    def analyze(self, market_data: MarketData) -> StrategyResult:
        """Called once per bar with point-in-time data only (no lookahead).
        Must return a StrategyResult; must not raise on ordinary inputs."""
        raise NotImplementedError

    def fit(self, train_data: pd.DataFrame) -> None:
        """Optional hook for strategies with tunable parameters.

        Contract: the backtester calls this ONLY with the train split.
        A strategy must never load its own historical data independently
        for tuning purposes — that's how lookahead bias sneaks in.
        Default: no-op, for strategies with no fittable parameters.
        """
        return None
