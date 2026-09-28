import pandas as pd
from ..core.models import Regime
from ..indicators.momentum import sma

class RegimeDetector:
    def detect(self, history: pd.DataFrame) -> Regime:
        if len(history) < 50: return Regime.UNKNOWN
        fast, slow = sma(history.close,20).iloc[-1], sma(history.close,50).iloc[-1]
        if pd.isna(fast) or pd.isna(slow): return Regime.UNKNOWN
        return Regime.TRENDING_UP if fast > slow else Regime.TRENDING_DOWN
