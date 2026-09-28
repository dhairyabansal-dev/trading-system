from __future__ import annotations

from dataclasses import dataclass, field

from ..core.models import Regime, Signal, StrategyResult


class Combiner:
    def combine(
        self, results: list[StrategyResult], regime: Regime | None = None
    ) -> StrategyResult:
        raise NotImplementedError


class PassThroughCombiner(Combiner):
    def combine(self, results: list[StrategyResult], regime: Regime | None = None) -> StrategyResult:
        if len(results) != 1:
            raise ValueError("PassThroughCombiner requires exactly one strategy")
        return results[0]


@dataclass
class StrategyTrackRecord:
    strategy_name: str
    oos_trades: int = 0
    validated: bool = False
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    max_drawdown: float = 0.0
    decay: float = 0.0
    regime_affinity: dict[str, float] = field(default_factory=dict)


@dataclass
class WeightedVoteCombiner(Combiner):
    records: dict[str, StrategyTrackRecord] = field(default_factory=dict)
    min_oos_trades: int = 30
    min_oos_sharpe: float = 0.0
    neutral_regime_multiplier: float = 1.0

    def combine(
        self, results: list[StrategyResult], regime: Regime | None = None
    ) -> StrategyResult:
        if not results:
            return StrategyResult("ultimate_combiner", Signal.HOLD, 0.0, 0.0, {"reason": "no results"})

        contributors: list[StrategyResult] = []
        weights: list[float] = []

        for result in results:
            record = self.records.get(result.strategy_name)
            if record is None or not record.validated:
                continue
            if record.oos_trades < self.min_oos_trades or record.oos_sharpe < self.min_oos_sharpe:
                continue

            performance = max(0.0, 1.0 + record.oos_sharpe)
            stability = max(0.0, 1.0 - min(1.0, max(0.0, record.max_drawdown)))
            freshness = max(0.0, 1.0 - min(1.0, max(0.0, record.decay)))

            affinity = self.neutral_regime_multiplier
            if regime is not None and record.regime_affinity:
                affinity = max(0.0, record.regime_affinity.get(regime.value, self.neutral_regime_multiplier))

            weight = result.confidence * performance * stability * freshness * affinity
            if weight > 0 and result.signal != Signal.HOLD:
                contributors.append(result)
                weights.append(weight)

        if not contributors:
            return StrategyResult(
                "ultimate_combiner", Signal.HOLD, 0.0, 0.0,
                {"reason": "no validated contributors"},
            )

        denominator = sum(weights)
        score = sum(r.score * w for r, w in zip(contributors, weights)) / denominator
        signal = Signal.BUY if score > 0.10 else Signal.SELL if score < -0.10 else Signal.HOLD
        confidence = min(1.0, denominator / max(1.0, len(results)))

        return StrategyResult(
            "ultimate_combiner",
            signal,
            score,
            confidence,
            {
                "contributors": [r.strategy_name for r in contributors],
                "regime": regime.value if regime else None,
                "weights": {r.strategy_name: round(w / denominator, 4) for r, w in zip(contributors, weights)},
            },
        )
