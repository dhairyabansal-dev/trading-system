import pandas as pd

from ultimate_trading.core.models import MarketData, Regime, Signal
from ultimate_trading.data.synthetic import generate_ohlcv
from ultimate_trading.indicators.technical import atr, rsi, zscore
from ultimate_trading.regime.detector import RegimeDetector
from ultimate_trading.strategies.breakout.donchian import DonchianBreakoutStrategy
from ultimate_trading.strategies.mean_reversion.zscore import ZScoreMeanReversionStrategy
from ultimate_trading.strategies.trend.sma import SMATrendStrategy
from ultimate_trading.strategies.pattern.engulfing import EngulfingPatternStrategy
from ultimate_trading.strategies.orderflow.volume_delta import VolumeDeltaProxyStrategy
from ultimate_trading.strategies.rules.rsi_trend import RSITrendRuleStrategy


def md(data):
    row = data.iloc[-1]
    return MarketData("TEST", data.index[-1].to_pydatetime(), *map(float, row), data)


def test_indicators_are_finite_after_warmup():
    data = generate_ohlcv(120)
    assert pd.notna(atr(data).iloc[-1])
    assert pd.notna(rsi(data["close"]).iloc[-1])
    assert pd.notna(zscore(data["close"]).iloc[-1])


def test_regime_detector_returns_known_regime_after_warmup():
    data = generate_ohlcv(160, seed=11)
    regime = RegimeDetector().detect(data)
    assert regime in set(Regime)


def test_strategy_family_contracts():
    data = generate_ohlcv(180, seed=17)
    for strategy in (
        DonchianBreakoutStrategy(),
        ZScoreMeanReversionStrategy(),
        SMATrendStrategy(),
        EngulfingPatternStrategy(),
        VolumeDeltaProxyStrategy(),
        RSITrendRuleStrategy(),
    ):
        result = strategy.analyze(md(data))
        assert result.signal in Signal
        assert -1 <= result.score <= 1
        assert 0 <= result.confidence <= 1


def test_donchian_uses_prior_channel_not_current_high():
    data = generate_ohlcv(80, seed=3)
    current_high_before = float(data["high"].iloc[-1])
    data.iloc[-1, data.columns.get_loc("high")] = current_high_before * 10.0
    data.iloc[-1, data.columns.get_loc("close")] = current_high_before * 2.0
    result = DonchianBreakoutStrategy(window=20).analyze(md(data))
    assert result.metadata["donchian_upper"] < data["high"].iloc[-1]


def test_regime_affinity_changes_ensemble_weight():
    from ultimate_trading.ensemble.combiner import StrategyTrackRecord, WeightedVoteCombiner
    from ultimate_trading.core.models import StrategyResult

    combiner = WeightedVoteCombiner(
        records={
            "trend": StrategyTrackRecord(
                "trend", oos_trades=50, validated=True, oos_sharpe=1.0,
                regime_affinity={Regime.TRENDING_UP.value: 2.0, Regime.RANGING.value: 0.2},
            ),
            "mean": StrategyTrackRecord(
                "mean", oos_trades=50, validated=True, oos_sharpe=1.0,
                regime_affinity={Regime.TRENDING_UP.value: 0.2, Regime.RANGING.value: 2.0},
            ),
        }
    )
    results = [
        StrategyResult("trend", Signal.BUY, 0.8, 1.0),
        StrategyResult("mean", Signal.SELL, -0.8, 1.0),
    ]
    trending = combiner.combine(results, Regime.TRENDING_UP)
    ranging = combiner.combine(results, Regime.RANGING)
    assert trending.metadata["weights"]["trend"] > trending.metadata["weights"]["mean"]
    assert ranging.metadata["weights"]["mean"] > ranging.metadata["weights"]["trend"]
