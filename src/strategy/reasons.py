"""低位说明。只描述价格位置，不编造下跌原因。"""
from __future__ import annotations

from typing import Any


def _fmt(v: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        if v is None:
            return "暂缺"
        fv = float(v)
        if fv != fv:  # NaN
            return "暂缺"
        return f"{fv:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "暂缺"


def build_reason(
    metrics: dict[str, Any],
    rr_info: dict[str, Any] | None = None,
    fund: dict[str, Any] | None = None,
) -> str:
    del rr_info, fund
    pos = metrics.get("range_pos")
    pos_pct = None
    try:
        if pos is not None and float(pos) == float(pos):
            pos_pct = float(pos) * 100
    except (TypeError, ValueError):
        pos_pct = None
    return (
        f"近一年自高点回撤约{_fmt(metrics.get('drawdown_250'))}%，"
        f"收盘处在这段高低区间从低点算起的{_fmt(pos_pct)}%位置。"
    )
