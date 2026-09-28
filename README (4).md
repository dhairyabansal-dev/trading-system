# market_making

Not implemented. Placeholder directory per docs/ARCHITECTURE.md's tree.

Implement against `strategies.base.Strategy` (see `strategies/momentum/simple_momentum.py`
for the reference implementation). Before wiring into `ensemble.WeightedVoteCombiner`,
it needs a walk-forward out-of-sample validation run, same as every other strategy —
see docs/ARCHITECTURE.md §1.
