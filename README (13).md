# ultimate-trading-system

Research / paper-trading system. **Not** an auto-execution system — see
`docs/ARCHITECTURE.md` for the locked Milestone 0 spec, and read it before
adding a second strategy. The short version: the biggest risk here is
overfitting the ensemble to the backtest window, not "not enough
strategies." Every guardrail below exists because of that.

## Status: Milestone 0 (vertical slice)

One strategy (`simple_momentum`), one combiner (`PassThroughCombiner`),
running end-to-end through a walk-forward backtest on synthetic data. This
validates the interfaces, not any trading edge. Do not read the metrics as
"this strategy works."

## Quickstart

```bash
pip install -r requirements.txt
python run_milestone0.py
```

Expected output: an in-sample / out-of-sample metrics table plus a
reminder banner. If in-sample dramatically outperforms out-of-sample,
that's the overfitting risk showing up as intended — it's a signal to
investigate, not a bug to silence.

Run tests:

```bash
pytest tests/
```

## Layout

```
core/            shared contracts, data, indicators, regime, risk, portfolio
strategies/      one subfolder per strategy family (mechanism-named, not vendor-named)
ensemble/        combiner interface + PassThroughCombiner + WeightedVoteCombiner
backtesting/     walk-forward engine, the only place a fit() sees train-only data
configs/         per-milestone YAML configs
tests/           unit tests — treat test additions as part of "implement," not optional
docs/            ARCHITECTURE.md (locked spec), add milestone docs here as scope grows
```

## Before adding strategy #2

Read `docs/ARCHITECTURE.md` §1. In short:
- It needs its own out-of-sample validation before it gets any weight in
  the combiner (see `WeightedVoteCombiner.register_track_record`).
- Its `fit()` must only ever be called with train-split data — never wire
  it up to load its own historical data.
- Adding it should come with an out-of-sample metrics comparison against
  the single-strategy baseline, not just an in-sample "it's an improvement."

## What "Pionex module," "Tickeron module," etc. mean here

They're mechanism names, not vendor names: "Pionex" → grid strategy engine,
"Tickeron" → pattern recognition engine, "Hummingbot" → market
making / order-flow engine, "Coinrule" → rule-based strategy engine. We're
borrowing mechanical concepts, not recreating the platforms.
