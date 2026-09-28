from abc import ABC, abstractmethod
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("OHLCV index must be a DatetimeIndex")
    if data.index.tz is None:
        raise ValueError("OHLCV index must be timezone-aware")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("OHLCV index must be increasing and unique")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("OHLCV data contains missing values")
    if (data["high"] < data[["open", "close"]].max(axis=1)).any():
        raise ValueError("high is below open/close")
    if (data["low"] > data[["open", "close"]].min(axis=1)).any():
        raise ValueError("low is above open/close")
    if (data["low"] <= 0).any() or (data["close"] <= 0).any():
        raise ValueError("prices must be positive")
    return data.copy()


class MarketDataLoader(ABC):
    @abstractmethod
    def load(self, symbol: str, start=None, end=None, timeframe: str = "15m") -> pd.DataFrame:
        raise NotImplementedError
