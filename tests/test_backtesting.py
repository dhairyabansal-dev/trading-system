import pandas as pd
from ultimate_trading.backtesting.engine import WalkForwardSplitter
def test_split():
    s=WalkForwardSplitter(.7).split(pd.DataFrame({"x":range(10)}));assert len(s.train)==7 and len(s.test)==3
