from dataclasses import dataclass
from datetime import datetime, timezone
from .core.models import Signal, StrategyResult


@dataclass(frozen=True)
class TradingViewSignal:
    symbol: str
    timestamp: datetime
    action: Signal
    score: float
    confidence: float
    reason: str


def to_tradingview_signal(symbol: str, timestamp: datetime, result: StrategyResult) -> TradingViewSignal:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return TradingViewSignal(
        symbol=symbol,
        timestamp=timestamp,
        action=result.signal,
        score=result.score,
        confidence=result.confidence,
        reason=result.metadata.get("reason", result.strategy_name),
    )


def to_webhook_payload(signal: TradingViewSignal) -> dict:
    return {
        "symbol": signal.symbol,
        "timestamp": signal.timestamp.isoformat(),
        "action": signal.action.value,
        "score": round(signal.score, 6),
        "confidence": round(signal.confidence, 6),
        "reason": signal.reason,
    }
