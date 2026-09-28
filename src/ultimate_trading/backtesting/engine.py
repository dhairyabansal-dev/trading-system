from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.models import MarketData, Signal
from ..core.strategy import Strategy
from ..data.loader import validate_ohlcv
from ..regime.detector import RegimeDetector


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


class WalkForwardSplitter:
    def __init__(self, train_fraction: float = 0.7):
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        self.train_fraction = train_fraction

    def split(self, data: pd.DataFrame) -> Split:
        data = validate_ohlcv(data)
        i = int(len(data) * self.train_fraction)
        if i < 2 or len(data) - i < 2:
            raise ValueError("dataset is too small for train/test split")
        return Split(data.iloc[:i].copy(), data.iloc[i:].copy())


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    max_position_fraction: float = 1.0

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("costs cannot be negative")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be in (0, 1]")


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    trades: tuple[Trade, ...]
    total_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    profit_factor: float

    @property
    def trade_count(self) -> int:
        return len(self.trades)


class BacktestEngine:
    """Event-driven candle backtester with next-open execution.

    A signal is generated from information available through bar t and the
    order is filled at bar t+1 open. This is the structural look-ahead guard.
    """

    def __init__(self, config: BacktestConfig | None = None, splitter=None):
        self.config = config or BacktestConfig()
        self.splitter = splitter or WalkForwardSplitter()
        self.regime_detector = RegimeDetector()

    def prepare(self, data: pd.DataFrame) -> Split:
        return self.splitter.split(data)

    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "SYNTH",
        train_fraction: float | None = None,
    ) -> BacktestResult:
        data = validate_ohlcv(data)
        if train_fraction is not None:
            split = WalkForwardSplitter(train_fraction).split(data)
            strategy.fit(split.train.copy())
            # Preserve the training history as read-only context for indicator
            # warm-up, while only bars in the test interval can create trades.
            test = split.test
            context_start = max(0, len(split.train) - 250)
            data = pd.concat([split.train.iloc[context_start:], test])
            return self._run_data(data, strategy, symbol, trade_start=test.index[0])

        return self._run_data(data, strategy, symbol)

    def run_ensemble(
        self,
        data: pd.DataFrame,
        strategies: list[Strategy],
        combiner,
        symbol: str = "SYNTH",
        train_fraction: float | None = None,
        regime_detector: RegimeDetector | None = None,
    ) -> BacktestResult:
        if not strategies:
            raise ValueError("at least one strategy is required")
        data = validate_ohlcv(data)
        trade_start = None
        if train_fraction is not None:
            split = WalkForwardSplitter(train_fraction).split(data)
            for strategy in strategies:
                strategy.fit(split.train.copy())
            trade_start = split.test.index[0]
            context_start = max(0, len(split.train) - 250)
            data = pd.concat([split.train.iloc[context_start:], split.test])

        detector = regime_detector or self.regime_detector

        def ensemble_signal(md: MarketData) -> object:
            regime = detector.detect(md.history)
            results = [strategy.analyze(md) for strategy in strategies]
            return combiner.combine(results, regime=regime)

        return self._run_data(data, ensemble_signal, symbol, trade_start=trade_start)

    def _run_data(self, data: pd.DataFrame, strategy_or_fn, symbol: str, trade_start=None) -> BacktestResult:
        data = validate_ohlcv(data)
        equity = self.config.initial_capital
        peak = equity
        equity_values: list[tuple[pd.Timestamp, float]] = []
        trades: list[Trade] = []
        position = 0
        quantity = 0.0
        entry_price = 0.0
        entry_time = None

        for i in range(len(data) - 1):
            row = data.iloc[i]
            history = data.iloc[: i + 1]
            md = MarketData(
                symbol=symbol,
                timestamp=data.index[i],
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                history=history,
            )
            result = strategy_or_fn(md) if callable(strategy_or_fn) else strategy_or_fn.analyze(md)
            next_time = data.index[i + 1]
            if trade_start is not None and next_time < trade_start:
                continue

            desired = 1 if result.signal == Signal.BUY else -1 if result.signal == Signal.SELL else 0
            if result.confidence <= 0:
                desired = 0

            if position != 0 and desired != position:
                exit_price = self._adjust_price(float(data.iloc[i + 1].open), -position)
                gross = position * quantity * (exit_price - entry_price)
                costs = self._cost(entry_price * quantity) + self._cost(exit_price * quantity)
                net = gross - costs
                equity += net
                trades.append(self._trade(entry_time, next_time, position, entry_price, exit_price, quantity, gross, costs, net))
                position = 0
                quantity = 0.0

            if position == 0 and desired != 0:
                notional = equity * self.config.max_position_fraction * min(1.0, result.confidence)
                entry_price = self._adjust_price(float(data.iloc[i + 1].open), desired)
                quantity = notional / entry_price
                position = desired
                entry_time = next_time

            mark = float(data.iloc[i + 1].close)
            unrealized = position * quantity * (mark - entry_price) if position else 0.0
            marked_equity = equity + unrealized
            peak = max(peak, marked_equity)
            equity_values.append((next_time, marked_equity))

        if position:
            final_time = data.index[-1]
            exit_price = self._adjust_price(float(data.iloc[-1].close), -position)
            gross = position * quantity * (exit_price - entry_price)
            costs = self._cost(entry_price * quantity) + self._cost(exit_price * quantity)
            net = gross - costs
            equity += net
            trades.append(self._trade(entry_time, final_time, position, entry_price, exit_price, quantity, gross, costs, net))
            if equity_values:
                equity_values[-1] = (final_time, equity)

        curve = pd.Series(dict(equity_values), dtype=float)
        if curve.empty:
            curve = pd.Series([self.config.initial_capital], index=[data.index[-1]], dtype=float)

        returns = curve.pct_change().dropna()
        running_peak = curve.cummax()
        drawdown = curve / running_peak - 1.0
        max_dd = float(abs(drawdown.min())) if not drawdown.empty else 0.0
        sharpe = float((returns.mean() / returns.std()) * (252 ** 0.5)) if len(returns) > 1 and returns.std() > 0 else 0.0
        wins = [t.net_pnl for t in trades if t.net_pnl > 0]
        losses = [-t.net_pnl for t in trades if t.net_pnl < 0]
        profit_factor = float(sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)

        return BacktestResult(
            equity_curve=curve,
            trades=tuple(trades),
            total_return=float(equity / self.config.initial_capital - 1.0),
            max_drawdown=max_dd,
            sharpe=sharpe,
            win_rate=float(len(wins) / len(trades)) if trades else 0.0,
            profit_factor=profit_factor,
        )

    @staticmethod
    def _trade(entry_time, exit_time, side, entry_price, exit_price, quantity, gross, costs, net) -> Trade:
        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            gross_pnl=gross,
            costs=costs,
            net_pnl=net,
            return_pct=net / (abs(entry_price * quantity) or 1.0),
        )

    def _cost(self, notional: float) -> float:
        return notional * (self.config.fee_bps / 10_000)

    def _adjust_price(self, price: float, direction: int) -> float:
        slip = self.config.slippage_bps / 10_000
        return price * (1 + slip if direction > 0 else 1 - slip)
