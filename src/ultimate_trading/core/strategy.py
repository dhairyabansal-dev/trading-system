from abc import ABC, abstractmethod
import pandas as pd
from .models import MarketData, StrategyResult

class Strategy(ABC):
    name: str = "unnamed_strategy"
    @abstractmethod
    def analyze(self, market_data: MarketData) -> StrategyResult:
        raise NotImplementedError
    def fit(self, train_data: pd.DataFrame) -> None:
        return None
