import numpy as np
import pandas as pd

def generate_ohlcv(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[100, close[:-1]]
    spread = np.abs(rng.normal(0, .004, n_bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(10, .4, n_bars)
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="15min", tz="UTC")
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":volume}, index=idx)
