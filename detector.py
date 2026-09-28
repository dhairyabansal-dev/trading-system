"""
Regime detection — deliberately crude in Milestone 0.

Treat this with the same suspicion as any other strategy: it is fit on
data and can overfit just like the trading strategies can. It exists here
mainly so the interface (MarketData -> Regime) is fixed before more
sophisticated detectors are swapped in.
"""
from __future__ import annotations

import pandas as pd

from core.contracts import Regime
from core.indicators.momentum import rolling_volatility, simple_moving_average


def detect_regime(
    history: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    vol_window: int = 20,
    vol_ranging_threshold: float = 0.9,
) -> Regime:
    """Very simple SMA-crossover + volatility heuristic.

    - If we don't have enough history yet, return UNKNOWN rather than
      guessing — strategies should treat UNKNOWN as "no regime edge,"
      not as a stand-in for RANGING.
    - Trend direction from fast vs slow SMA.
    - If recent volatility is elevated relative to its own trailing average,
      call it RANGING regardless of the SMA cross (choppy, not trending).
    """
    close = history["close"]
    if len(close) < slow_window + vol_window:
        return Regime.UNKNOWN

    fast = simple_moving_average(close, fast_window).iloc[-1]
    slow = simple_moving_average(close, slow_window).iloc[-1]
    vol = rolling_volatility(close, vol_window)
    recent_vol = vol.iloc[-1]
    trailing_vol_avg = vol.iloc[-(vol_window * 3):].mean() if len(vol.dropna()) >= vol_window * 3 else vol.mean()

    if pd.isna(fast) or pd.isna(slow) or pd.isna(recent_vol) or pd.isna(trailing_vol_avg):
        return Regime.UNKNOWN

    if trailing_vol_avg > 0 and recent_vol > trailing_vol_avg / max(vol_ranging_threshold, 1e-9):
        return Regime.RANGING

    return Regime.TRENDING_UP if fast > slow else Regime.TRENDING_DOWN
