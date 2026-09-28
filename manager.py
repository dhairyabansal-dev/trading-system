"""
Risk layer. Converts a StrategyResult (or combiner output) into a bounded
position sizing decision. This never talks to a broker/exchange — it only
produces a target fraction of paper-trading equity for the portfolio
tracker to simulate.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.contracts import RiskAdjustedSignal, Signal, StrategyResult


@dataclass(frozen=True)
class RiskConfig:
    max_position_fraction: float = 0.25   # never commit more than this fraction of equity
    min_confidence_to_act: float = 0.2    # below this, force HOLD regardless of score
    max_drawdown_halt: float = 0.25       # if paper equity draws down more than this, halt new entries


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def size(self, result: StrategyResult, current_drawdown: float) -> RiskAdjustedSignal:
        if current_drawdown >= self.config.max_drawdown_halt:
            return RiskAdjustedSignal(
                signal=Signal.HOLD,
                target_position_fraction=0.0,
                reason=f"drawdown halt triggered ({current_drawdown:.1%} >= "
                       f"{self.config.max_drawdown_halt:.1%})",
            )

        if result.confidence < self.config.min_confidence_to_act:
            return RiskAdjustedSignal(
                signal=Signal.HOLD,
                target_position_fraction=0.0,
                reason=f"confidence {result.confidence:.2f} below floor "
                       f"{self.config.min_confidence_to_act:.2f}",
            )

        # Position size scales with |score| * confidence, capped at max_position_fraction.
        raw_fraction = abs(result.score) * result.confidence
        fraction = min(raw_fraction, self.config.max_position_fraction)

        return RiskAdjustedSignal(
            signal=result.signal,
            target_position_fraction=fraction if result.signal != Signal.HOLD else 0.0,
            reason=f"sized from score={result.score:.2f}, confidence={result.confidence:.2f}",
        )
