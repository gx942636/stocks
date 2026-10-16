"""中线 / 短线切换。短线参数独立，拉日线仍取更长的那段历史。"""
from __future__ import annotations

from typing import Any


def resolve_style(config: dict[str, Any] | None) -> str:
    style = str((config or {}).get("style") or "mid").strip().lower()
    return "short" if style == "short" else "mid"


def _block(config: dict[str, Any], key: str) -> dict[str, Any]:
    raw = config.get(key) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def active_strategy(config: dict[str, Any] | None) -> dict[str, Any]:
    """当前要用来筛选的那一套参数。"""
    cfg = config or {}
    if resolve_style(cfg) == "short":
        return _block(cfg, "strategy_short")
    return _block(cfg, "strategy")


def fetch_strategy(config: dict[str, Any] | None) -> dict[str, Any]:
    """拉取用当前策略的股票池门槛，历史长度取中线和短线里更长的一段。"""
    cfg = config or {}
    mid = _block(cfg, "strategy")
    short = _block(cfg, "strategy_short")
    active = active_strategy(cfg)
    history = max(
        int(mid.get("history_calendar_days", 140) or 140),
        int(short.get("history_calendar_days", 80) or 80),
    )
    min_bars = max(
        int(mid.get("pullback_days", 60) or 60),
        int(short.get("min_bars", 21) or 21),
    )
    active["history_calendar_days"] = history
    active["pullback_days"] = min_bars
    return active
