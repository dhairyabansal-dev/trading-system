"""
Ensemble layer. Combines one or more StrategyResults into one.

Milestone 0 rule (see docs/ARCHITECTURE.md §1): a strategy earns a combiner
seat by validated out-of-sample performance; it isn't given one by default.
PassThroughCombiner is what's actually wired up for the one-strategy slice.
WeightedVoteCombiner is the interface for later, deliberately built to
refuse to silently trust an unvalidated strategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.contracts import Signal, StrategyResult


class Combiner(ABC):
    @abstractmethod
    def combine(self, results: list[StrategyResult]) -> StrategyResult:
        raise NotImplementedError


class PassThroughCombiner(Combiner):
    """For exactly one strategy. Raises if handed more than one result —
    this is intentional: don't let a second strategy get silently voted
    through a combiner that was never designed to weigh it."""

    def combine(self, results: list[StrategyResult]) -> StrategyResult:
        if len(results) != 1:
            raise ValueError(
                f"PassThroughCombiner expects exactly 1 strategy result, got {len(results)}. "
                "Use WeightedVoteCombiner once you have validated out-of-sample "
                "track records for multiple strategies."
            )
        return results[0]


@dataclass
class StrategyTrackRecord:
    strategy_name: str
    out_of_sample_trades: int = 0
    out_of_sample_win_rate: float = 0.0
    validated: bool = False


@dataclass
class WeightedVoteCombiner(Combiner):
    """Confidence-weighted vote across multiple strategies, gated by
    out-of-sample track record so an unvalidated strategy can't move the
    combined signal.

    NOT wired up to real strategies in Milestone 0 — the vertical slice
    only has one strategy, and PassThroughCombiner is used for it. This
    class exists so the interface is fixed and testable ahead of scaling
    to more strategies.
    """
    track_records: dict[str, StrategyTrackRecord] = field(default_factory=dict)
    min_oos_trades: int = 30  # a strategy needs at least this many OOS trades before its vote counts

    def register_track_record(self, record: StrategyTrackRecord) -> None:
        record.validated = record.out_of_sample_trades >= self.min_oos_trades
        self.track_records[record.strategy_name] = record

    def _weight_for(self, result: StrategyResult) -> float:
        record = self.track_records.get(result.strategy_name)
        if record is None or not record.validated:
            return 0.0  # unvalidated strategies get zero weight, not a small default weight
        # Weight by confidence and by how much out-of-sample evidence exists,
        # not just by in-sample-flavored confidence alone.
        evidence_factor = min(1.0, record.out_of_sample_trades / (self.min_oos_trades * 3))
        return result.confidence * evidence_factor

    def combine(self, results: list[StrategyResult]) -> StrategyResult:
        weighted_scores = []
        total_weight = 0.0
        for r in results:
            w = self._weight_for(r)
            weighted_scores.append(w * r.score)
            total_weight += w

        if total_weight == 0.0:
            return StrategyResult(
                strategy_name="ultimate_combiner",
                signal=Signal.HOLD,
                score=0.0,
                confidence=0.0,
                metadata={"reason": "no validated strategies contributed weight"},
            )

        combined_score = sum(weighted_scores) / total_weight
        combined_confidence = min(1.0, total_weight / len(results))
        signal = Signal.BUY if combined_score > 0.1 else Signal.SELL if combined_score < -0.1 else Signal.HOLD

        return StrategyResult(
            strategy_name="ultimate_combiner",
            signal=signal,
            score=combined_score,
            confidence=combined_confidence,
            metadata={"n_contributing": sum(1 for r in results if self._weight_for(r) > 0)},
        )
