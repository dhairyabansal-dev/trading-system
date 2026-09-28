"""
Paper-trading portfolio tracker. Simulates equity and a single position in
one symbol against a target fraction — no broker/exchange integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.contracts import RiskAdjustedSignal, Signal


@dataclass
class PortfolioState:
    cash: float
    position_units: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    peak_equity: float = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.position_units * price

    def drawdown(self, price: float) -> float:
        eq = self.equity(price)
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - eq) / self.peak_equity)


class PaperPortfolio:
    def __init__(self, starting_cash: float = 10_000.0):
        self.state = PortfolioState(cash=starting_cash, peak_equity=starting_cash)

    def current_drawdown(self, price: float) -> float:
        return self.state.drawdown(price)

    def apply(self, decision: RiskAdjustedSignal, price: float) -> None:
        """Rebalance the simulated position toward the target fraction of
        current equity. BUY/SELL both just mean 'move toward this target
        fraction, long or short' — Milestone 0 keeps this simple (no
        leverage, no shorting) by clamping fraction to [0, max]."""
        equity = self.state.equity(price)
        target_fraction = decision.target_position_fraction
        if decision.signal == Signal.SELL:
            target_fraction = 0.0  # Milestone 0: no shorting, SELL means flatten.
        elif decision.signal == Signal.HOLD:
            target_fraction = self.state.position_units * price / equity if equity > 0 else 0.0

        target_value = equity * target_fraction
        target_units = target_value / price if price > 0 else 0.0
        delta_units = target_units - self.state.position_units

        self.state.cash -= delta_units * price
        self.state.position_units = target_units
        self.state.equity_curve.append(self.state.equity(price))
