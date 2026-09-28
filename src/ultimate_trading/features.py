import pandas as pd

from .indicators.momentum import rate_of_change, rolling_volatility, sma


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    close = out["close"]
    out["roc_5"] = rate_of_change(close, 5)
    out["roc_20"] = rate_of_change(close, 20)
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["volatility_20"] = rolling_volatility(close, 20)
    out["range_pct"] = (out["high"] - out["low"]) / close
    out["volume_zscore_20"] = (
        (out["volume"] - out["volume"].rolling(20).mean())
        / out["volume"].rolling(20).std()
    )
    return out
