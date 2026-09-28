"""Minimal indicator set for the Milestone 0 momentum slice.

Kept deliberately small — this is proving the interface, not building an
indicator library.
"""
from __future__ import annotations

import pandas as pd


def rate_of_change(close: pd.Series, window: int) -> pd.Series:
    """(close_t / close_{t-window}) - 1"""
    return close.pct_change(periods=window)


def simple_moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean()


def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of simple returns — used by the regime detector."""
    returns = close.pct_change()
    return returns.rolling(window=window, min_periods=window).std()
