from dataclasses import dataclass,field
from ..core.models import Signal,StrategyResult
class Combiner:
    def combine(self,results:list[StrategyResult])->StrategyResult: raise NotImplementedError
class PassThroughCombiner(Combiner):
    def combine(self,results):
        if len(results)!=1: raise ValueError("PassThroughCombiner requires exactly one strategy")
        return results[0]
@dataclass
class StrategyTrackRecord:
    strategy_name:str
    oos_trades:int=0
    validated:bool=False
@dataclass
class WeightedVoteCombiner(Combiner):
    records:dict[str,StrategyTrackRecord]=field(default_factory=dict)
    min_oos_trades:int=30
    def combine(self,results):
        contributors=[]
        for r in results:
            rec=self.records.get(r.strategy_name)
            if rec and rec.validated and rec.oos_trades>=self.min_oos_trades: contributors.append(r)
        if not contributors:return StrategyResult("ultimate_combiner",Signal.HOLD,0,0,{"reason":"no validated contributors"})
        score=sum(r.score*r.confidence for r in contributors)/sum(r.confidence for r in contributors)
        signal=Signal.BUY if score>.1 else Signal.SELL if score<-.1 else Signal.HOLD
        return StrategyResult("ultimate_combiner",signal,score,min(1,sum(r.confidence for r in contributors)/len(results)),{"contributors":[r.strategy_name for r in contributors]})
