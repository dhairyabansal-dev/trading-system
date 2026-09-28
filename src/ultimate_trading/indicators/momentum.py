import pandas as pd

def rate_of_change(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window)

def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()

def rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change().rolling(window, min_periods=window).std()
