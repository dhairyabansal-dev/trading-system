from abc import ABC, abstractmethod
import pandas as pd

class MarketDataLoader(ABC):
    @abstractmethod
    def load(self, symbol: str, start=None, end=None, timeframe: str = "15m") -> pd.DataFrame:
        raise NotImplementedError
