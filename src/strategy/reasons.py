"""买入日说明。只描述价格结构，不编造下跌原因。"""
from __future__ import annotations

from typing import Any


def _fmt(v: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        if v is None:
            return "暂缺"
        fv = float(v)
        if fv != fv:
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
    return (
        f"相对近60日高点回撤{_fmt(metrics.get('drawdown_60'))}%，"
        f"收盘守在20日低点{_fmt(metrics.get('low_20'), 2)}元上方{_fmt(metrics.get('low_dist_pct'))}%，"
        f"近5日涨幅{_fmt(metrics.get('ret_5d'))}%，最近没有再创新低。"
    )
