from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    if window < 2:
        raise ValueError("RSI window must be >= 2")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out


def atr(history: pd.DataFrame, window: int = 14) -> pd.Series:
    if window < 1:
        raise ValueError("ATR window must be >= 1")
    high, low, close = history["high"], history["low"], history["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def bollinger_bands(
    close: pd.Series, window: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if window < 2 or n_std <= 0:
        raise ValueError("window must be >= 2 and n_std must be positive")
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    return middle + n_std * std, middle, middle - n_std * std


def zscore(close: pd.Series, window: int = 20) -> pd.Series:
    if window < 2:
        raise ValueError("z-score window must be >= 2")
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    return (close - mean) / std.replace(0, np.nan)


def donchian_channel(
    history: pd.DataFrame, window: int = 20
) -> tuple[pd.Series, pd.Series]:
    if window < 2:
        raise ValueError("Donchian window must be >= 2")
    return (
        history["high"].rolling(window, min_periods=window).max(),
        history["low"].rolling(window, min_periods=window).min(),
    )


def linreg_slope(close: pd.Series, window: int = 20) -> pd.Series:
    if window < 2:
        raise ValueError("slope window must be >= 2")

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        x = np.arange(len(values), dtype=float)
        return float(np.polyfit(x, values, 1)[0] / (np.mean(values) + 1e-12))

    return close.rolling(window, min_periods=window).apply(slope, raw=True)


def volume_delta(history: pd.DataFrame, window: int = 20) -> pd.Series:
    if window < 2:
        raise ValueError("volume-delta window must be >= 2")
    direction = np.sign(history["close"].diff()).fillna(0.0)
    signed_volume = direction * history["volume"]
    return signed_volume.rolling(window, min_periods=window).sum()


def bullish_engulfing(history: pd.DataFrame) -> pd.Series:
    o, c = history["open"], history["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    return (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)


def bearish_engulfing(history: pd.DataFrame) -> pd.Series:
    o, c = history["open"], history["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    return (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
