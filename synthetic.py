"""
Deterministic synthetic OHLCV generator.

This sandbox has no network access, so Milestone 0 validates the pipeline
on synthetic data rather than a real exchange feed. Swapping in real data
later means writing one function with this same return shape
(DataFrame indexed by timestamp, columns open/high/low/close/volume) —
nothing else in the system needs to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(
    n_bars: int = 1000,
    start_price: float = 100.0,
    annualized_drift: float = 0.05,
    annualized_vol: float = 0.6,
    bars_per_year: int = 365,
    seed: int = 42,
    freq: str = "1D",
    regime_shift_at: int | None = None,
    regime_shift_drift: float | None = None,
) -> pd.DataFrame:
    """Generate a reproducible synthetic price series via geometric Brownian
    motion, optionally with a drift regime shift partway through so the
    momentum strategy has something non-trivial to react to.

    Deterministic given `seed` — re-running produces byte-identical output,
    which matters for reproducible backtests.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year

    drifts = np.full(n_bars, annualized_drift)
    if regime_shift_at is not None:
        shift_drift = regime_shift_drift if regime_shift_drift is not None else -annualized_drift
        drifts[regime_shift_at:] = shift_drift

    shocks = rng.normal(0, 1, n_bars)
    log_returns = (drifts - 0.5 * annualized_vol**2) * dt + annualized_vol * np.sqrt(dt) * shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    # Build plausible OHLC around each close using small intrabar noise.
    intrabar_noise = rng.normal(0, annualized_vol * np.sqrt(dt) * 0.5, size=(n_bars, 2))
    high = close * (1 + np.abs(intrabar_noise[:, 0]))
    low = close * (1 - np.abs(intrabar_noise[:, 1]))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.lognormal(mean=10, sigma=0.5, size=n_bars)

    index = pd.date_range("2020-01-01", periods=n_bars, freq=freq)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    df.index.name = "timestamp"
    return df
