# Architecture Spec — Milestone 0

**Status: LOCKED for Milestone 0.** Changes require a new milestone doc, not silent edits here.

## 0. Scope of this milestone

This is a **research and paper-trading system**. There is no auto-execution,
no broker/exchange order routing, and no real capital at risk anywhere in
this milestone. The `execution/` and `alerts/` directories in the long-term
tree are placeholders only — they contain no code until a milestone
explicitly scopes them in, and that milestone must separately justify
graduating past paper trading.

Definition of done for Milestone 0:
- One strategy engine, implemented for real (not a stub), running against
  historical data through the backtester.
- The `Strategy` interface, `StrategyResult`/`MarketData` contracts, and the
  ensemble combiner interface are exercised end-to-end, even with only one
  strategy plugged in.
- A walk-forward train/test split exists in the backtester and is not
  bypassable by accident (see §3).
- No claim of profitability, edge, or "it works" is made anywhere in code
  comments, README, or output — only "it runs, and here's what the metrics
  say in-sample vs out-of-sample."

## 1. Why overfitting is the central risk, not a footnote

The instinct with a system like this is "more strategies = more signal."
That's backwards. Every strategy added is a new source of variance that the
combiner can latch onto in-sample and lose out-of-sample. Concretely:

- **N strategies voting is N chances to curve-fit the backtest window.**
  If you tune 10 strategies' parameters against the same historical window
  you'll backtest them on, you have effectively fit ~10x the free
  parameters to one dataset. The combiner then learns to trust whichever
  strategy got lucky on that window, not whichever strategy has real edge.
- **Ensembling doesn't fix this by default.** Averaging or voting reduces
  variance only if the underlying signals are diverse *and individually
  validated out-of-sample*. Averaging ten overfit strategies gives you an
  overfit average.
- **Regime detection is itself a strategy that can overfit.** Treat it with
  the same suspicion as everything downstream of it.

Concrete guardrails this architecture bakes in from Milestone 0:

1. **Walk-forward only.** `backtesting/` enforces a chronological
   train/validation/test split. A strategy is not allowed to see test-set
   data at fit time. This is enforced in code (`WalkForwardSplitter`), not
   left to discipline.
2. **A strategy earns a combiner seat; it isn't given one.** The
   `ensemble/` layer requires a minimum out-of-sample track record
   (`MIN_OOS_TRADES`, configurable) before a strategy's vote counts above
   zero weight. This is a stub in Milestone 0 but the interface is fixed
   now so it can't be an afterthought later.
3. **No parameter search against the full dataset.** Any hyperparameter
   tuning must run inside the train split only. The backtester raises if a
   strategy's `fit()` is called with test-split data.
4. **Report both curves.** Every backtest run emits in-sample and
   out-of-sample metrics side by side. A strategy/combiner change that
   improves in-sample and degrades out-of-sample is a regression, full
   stop, regardless of the in-sample number.
5. **Fewer, validated strategies beat many, unvalidated ones.** The
   combiner is built to work correctly with exactly one strategy. Adding
   the second, third, etc. is a deliberate, reviewed decision per strategy,
   not a default.

## 2. Layers and data contracts

```
MarketData  -->  Indicators  -->  Regime  -->  Strategy.analyze()
                                                     |
                                                     v
                                             StrategyResult
                                                     |
                                                     v
                                          Ensemble (Combiner)
                                                     |
                                                     v
                                            Risk-adjusted Signal
                                                     |
                                                     v
                                          Portfolio (paper only)
```

### 2.1 Core contracts (`core/contracts.py`)

```python
class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass(frozen=True)
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    # rolling window access, not just the single bar
    history: pd.DataFrame  # indexed by timestamp, OHLCV columns

@dataclass(frozen=True)
class StrategyResult:
    strategy_name: str
    signal: Signal
    score: float          # -1.0 .. 1.0, magnitude/direction of conviction
    confidence: float      # 0.0 .. 1.0, how much the strategy trusts itself
    metadata: dict         # free-form, for debugging/analysis, not logic
```

`score` and `confidence` are deliberately separate. `score` is "how
bullish/bearish," `confidence` is "how sure am I this regime suits me."
A trend strategy in a chop regime should report low confidence even with a
strong score — that separation is what lets the combiner discount a
strategy without the strategy having to know about other strategies.

### 2.2 Strategy interface (`strategies/base.py`)

```python
class Strategy(ABC):
    name: str

    @abstractmethod
    def analyze(self, market_data: MarketData) -> StrategyResult: ...

    def fit(self, train_data: pd.DataFrame) -> None:
        """Optional. Called ONLY with train-split data by the backtester."""
        return None
```

Every strategy family (grid, DCA, momentum, pattern, order-flow, etc.)
implements this one interface. Nothing downstream is allowed to special-case
a particular strategy family — that's what keeps 20 strategies pluggable
without rewrites.

### 2.3 Ensemble interface (`ensemble/`)

```python
class Combiner(ABC):
    @abstractmethod
    def combine(self, results: list[StrategyResult]) -> StrategyResult: ...
```

Milestone 0 ships `PassThroughCombiner` (1 strategy, no voting logic to
trust yet) and a `WeightedVoteCombiner` stub whose weights are driven by
each strategy's *validated out-of-sample* stats, not its in-sample stats or
a hand-picked constant.

### 2.4 Risk and portfolio (paper only)

`core/risk/` sizes positions and enforces limits (max position size, max
drawdown halt). `core/portfolio/` tracks a simulated paper-trading balance
and open positions. Neither talks to a broker/exchange API in this
milestone — there is no code path to a real order.

## 3. Backtesting contract

`backtesting/engine.py` owns the walk-forward split and is the *only*
component allowed to decide what data a strategy's `fit()` sees. Strategies
never load their own historical data for tuning purposes — this is a
structural guard against accidental lookahead/overfitting, not just a
convention.

Reported metrics per run: total return, Sharpe (naive, no risk-free
adjustment yet), max drawdown, win rate, trade count — each computed
separately for train and test splits.

## 4. What strategy names actually mean

Per the naming decision already made: "Pionex module" means a **grid
strategy engine**, "Tickeron module" means a **pattern-recognition engine**,
"Hummingbot module" means a **market-making/order-flow engine**, "Coinrule
module" means a **rule-based strategy engine**. We're taking mechanical
concepts, not recreating the platforms. `strategies/` is organized by
mechanism, not by vendor name.

## 5. Milestone 0 vertical slice

To validate the interfaces before scaling to 10+ strategies, Milestone 0
implements exactly one strategy end-to-end:

- `strategies/momentum/simple_momentum.py` — a rate-of-change momentum
  strategy, deliberately simple so the *plumbing* is what's being tested,
  not the strategy's edge.
- Synthetic OHLCV data generator (`core/data/synthetic.py`) — this sandbox
  has no network access to pull real market data; the slice runs on
  deterministic synthetic data so it's fully reproducible. Swapping in a
  real data source later means implementing one loader function, nothing
  else changes.
- `backtesting/engine.py` run end-to-end on that one strategy, with
  train/test split, producing the in-sample/out-of-sample report from §1.

This slice is intentionally boring. The goal is "does the interface hold
together," not "is momentum a good strategy."
