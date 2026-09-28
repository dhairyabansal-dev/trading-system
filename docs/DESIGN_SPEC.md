# Design Specification

## Purpose
Build a research-grade modular trading intelligence system where strategy mechanisms are developed independently, validated out-of-sample, and only then combined.

## Core pipeline
Market Data → Indicators → Regime → Strategy Engines → Ultimate Combiner → Risk → Paper Portfolio → Backtesting/Signals.

## Contracts
**MarketData:** point-in-time symbol, timestamp, OHLCV and history containing no future bars.

**StrategyResult:** strategy_name, signal (BUY/SELL/HOLD), score (-1..1), confidence (0..1), metadata.

Score and confidence are intentionally separate.

## Strategy interface
Every strategy implements `analyze(market_data) -> StrategyResult`. Optional `fit(train_data)` receives training data only through the backtester.

## Strategy families
Grid, DCA, momentum, breakout, mean reversion, pattern recognition, order flow, trend, rule-based and allocation/rebalancing. These are mechanisms, not copies of third-party platforms.

## Ultimate Combiner
The initial combiner is pass-through for a one-strategy vertical slice. Multi-strategy weighting will eventually consider validated out-of-sample performance, regime compatibility, confidence, signal correlation/diversity, recent degradation and transaction costs.

An unvalidated strategy must not silently receive ensemble weight.

## Regime detection
Regime detection is itself treated as a model and must be validated.

## Risk
Risk gates sit after combination and before paper portfolio application: exposure limits, confidence floor, drawdown halt and position sizing.

## Backtesting
The backtester owns chronological train/test splitting and is the only component allowed to decide what data `fit()` sees.

## Overfitting guardrails
1. Walk-forward validation.
2. Test data never enters fitting/tuning.
3. Parameter searches stay inside training windows.
4. Ensemble weights use out-of-sample evidence.
5. Correlated strategies are not treated as independent votes.
6. New strategies require incremental OOS evaluation.
7. Paper trading precedes any live execution research.
