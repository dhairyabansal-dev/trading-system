"""
Entrypoint for the Milestone 0 vertical slice: one strategy, paper trading,
walk-forward backtest. Run with:

    python run_milestone0.py

This proves the pipeline (data -> strategy -> combiner -> risk -> portfolio
-> metrics) holds together end-to-end. It says nothing about whether
momentum is a good strategy — see docs/ARCHITECTURE.md Milestone 0 scope.
"""
from __future__ import annotations

import yaml

from backtesting.engine import BacktestEngine, WalkForwardSplitter
from core.data.synthetic import generate_ohlcv
from core.risk.manager import RiskConfig, RiskManager
from ensemble.combiner import PassThroughCombiner
from strategies.momentum.simple_momentum import SimpleMomentumStrategy


def main(config_path: str = "configs/milestone0.yaml") -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    data = generate_ohlcv(
        n_bars=data_cfg["n_bars"],
        seed=data_cfg["seed"],
        regime_shift_at=data_cfg.get("regime_shift_at"),
    )

    strat_cfg = config["strategy"]
    strategy = SimpleMomentumStrategy(
        lookback=strat_cfg["lookback"],
        buy_threshold=strat_cfg["buy_threshold"],
        sell_threshold=strat_cfg["sell_threshold"],
        ranging_confidence_penalty=strat_cfg["ranging_confidence_penalty"],
    )

    risk_cfg = config["risk"]
    risk_manager = RiskManager(
        RiskConfig(
            max_position_fraction=risk_cfg["max_position_fraction"],
            min_confidence_to_act=risk_cfg["min_confidence_to_act"],
            max_drawdown_halt=risk_cfg["max_drawdown_halt"],
        )
    )

    bt_cfg = config["backtest"]
    engine = BacktestEngine(
        strategy=strategy,
        combiner=PassThroughCombiner(),
        risk_manager=risk_manager,
        starting_cash=bt_cfg["starting_cash"],
        splitter=WalkForwardSplitter(train_fraction=bt_cfg["train_fraction"]),
    )

    report = engine.run(symbol="SYNTH/USDT", data=data)

    print("=== Milestone 0: Simple Momentum, paper trading, walk-forward ===")
    print(f"Data: {len(data)} synthetic bars, seed={data_cfg['seed']}, "
          f"regime shift at bar {data_cfg.get('regime_shift_at')}\n")
    print(report.summary())
    print(
        "\nReminder: this is a paper-trading research slice on synthetic data. "
        "In-sample beating out-of-sample is a red flag, not a win; out-of-sample "
        "beating in-sample on 1 seed of synthetic data is not evidence of edge either."
    )


if __name__ == "__main__":
    main()
