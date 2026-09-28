"""
Backtesting engine. Owns the walk-forward train/test split — this is the
one place in the system allowed to decide what data a strategy's fit()
sees. See docs/ARCHITECTURE.md §1 and §3 for why this is structural, not
just a convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.contracts import MarketData, RiskAdjustedSignal
from core.portfolio.tracker import PaperPortfolio
from core.risk.manager import RiskManager
from ensemble.combiner import Combiner
from strategies.base import Strategy


@dataclass
class BacktestMetrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int

    def as_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "sharpe": round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "n_trades": self.n_trades,
        }


@dataclass
class BacktestReport:
    train: BacktestMetrics
    test: BacktestMetrics

    def summary(self) -> str:
        lines = ["Split       | Return  | Sharpe | MaxDD  | WinRate | Trades"]
        for label, m in (("train (IS)", self.train), ("test (OOS)", self.test)):
            lines.append(
                f"{label:<11} | {m.total_return:>6.2%} | {m.sharpe:>6.2f} | "
                f"{m.max_drawdown:>5.2%} | {m.win_rate:>6.2%} | {m.n_trades}"
            )
        return "\n".join(lines)


class WalkForwardSplitter:
    """Chronological split — never shuffles. `train_fraction` of the bars
    (by time order) go to train, the rest to test."""

    def __init__(self, train_fraction: float = 0.7):
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")
        self.train_fraction = train_fraction

    def split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(data) * self.train_fraction)
        return data.iloc[:split_idx], data.iloc[split_idx:]


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        combiner: Combiner,
        risk_manager: RiskManager,
        starting_cash: float = 10_000.0,
        splitter: WalkForwardSplitter | None = None,
    ):
        self.strategy = strategy
        self.combiner = combiner
        self.risk_manager = risk_manager
        self.starting_cash = starting_cash
        self.splitter = splitter or WalkForwardSplitter()

    def run(self, symbol: str, data: pd.DataFrame) -> BacktestReport:
        train_data, test_data = self.splitter.split(data)

        # Structural guard: fit() only ever sees the train split.
        self.strategy.fit(train_data.copy())

        train_metrics = self._run_split(symbol, train_data)
        test_metrics = self._run_split(symbol, test_data)
        return BacktestReport(train=train_metrics, test=test_metrics)

    def _run_split(self, symbol: str, data: pd.DataFrame) -> BacktestMetrics:
        portfolio = PaperPortfolio(starting_cash=self.starting_cash)
        n_trades = 0
        wins = 0
        prev_position_units = 0.0

        min_history = 2  # let indicators decide their own warm-up via HOLD/low-confidence output
        for i in range(min_history, len(data)):
            window = data.iloc[: i + 1]
            bar = data.iloc[i]
            md = MarketData(
                symbol=symbol,
                timestamp=data.index[i],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                history=window,
            )

            result = self.strategy.analyze(md)
            combined = self.combiner.combine([result])
            drawdown = portfolio.current_drawdown(bar["close"])
            decision: RiskAdjustedSignal = self.risk_manager.size(combined, drawdown)

            pre_units = portfolio.state.position_units
            portfolio.apply(decision, bar["close"])
            post_units = portfolio.state.position_units

            if pre_units != post_units:
                n_trades += 1
                # crude win/loss tally: did equity increase over this bar's trade?
                if len(portfolio.state.equity_curve) >= 2:
                    if portfolio.state.equity_curve[-1] > portfolio.state.equity_curve[-2]:
                        wins += 1
            prev_position_units = post_units

        return self._compute_metrics(portfolio, n_trades, wins)

    def _compute_metrics(self, portfolio: PaperPortfolio, n_trades: int, wins: int) -> BacktestMetrics:
        curve = portfolio.state.equity_curve
        if len(curve) < 2:
            return BacktestMetrics(0.0, 0.0, 0.0, 0.0, n_trades)

        curve = np.array(curve)
        total_return = curve[-1] / curve[0] - 1.0

        returns = np.diff(curve) / curve[:-1]
        sharpe = 0.0
        if returns.std() > 1e-12:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)

        running_peak = np.maximum.accumulate(curve)
        drawdowns = (running_peak - curve) / running_peak
        max_drawdown = float(np.max(drawdowns))

        win_rate = wins / n_trades if n_trades > 0 else 0.0

        return BacktestMetrics(
            total_return=float(total_return),
            sharpe=float(sharpe),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            n_trades=n_trades,
        )
