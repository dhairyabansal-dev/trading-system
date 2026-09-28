from dataclasses import dataclass
import pandas as pd
@dataclass(frozen=True)
class Split:
    train:pd.DataFrame
    test:pd.DataFrame
class WalkForwardSplitter:
    def __init__(self,train_fraction=.7):
        if not 0<train_fraction<1:raise ValueError("train_fraction must be between 0 and 1")
        self.train_fraction=train_fraction
    def split(self,data):
        i=int(len(data)*self.train_fraction)
        return Split(data.iloc[:i],data.iloc[i:])
class BacktestEngine:
    def __init__(self,splitter=None):self.splitter=splitter or WalkForwardSplitter()
    def prepare(self,data):return self.splitter.split(data)
