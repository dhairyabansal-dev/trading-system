from dataclasses import asdict

import pandas as pd

from .backtesting.engine import BacktestEngine, BacktestResult
from .core.strategy import Strategy


def summarize(result: BacktestResult) -> dict:
    return {
        "trade_count": result.trade_count,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "sharpe": result.sharpe,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
    }


def walk_forward(
    data: pd.DataFrame,
    strategy: Strategy,
    train_fraction: float = 0.7,
    engine: BacktestEngine | None = None,
) -> tuple[BacktestResult, dict]:
    engine = engine or BacktestEngine()
    result = engine.run(data, strategy, train_fraction=train_fraction)
    return result, summarize(result)
