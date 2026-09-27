"""推荐理由生成。"""
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
    rr_info: dict[str, Any],
    fund: dict[str, Any] | None = None,
) -> str:
    fund = fund or {}
    parts = [
        f"近60日回撤约{_fmt(metrics.get('drawdown_60d'))}%，"
        f"60日涨跌{_fmt(metrics.get('ret_60d'))}%，"
        f"近20日无涨停且量比{_fmt(metrics.get('volume_ratio'))}，处于冷却埋伏区",
        f"按回撤修复目标测算盈亏比约{_fmt(rr_info.get('rr'), 2)}，"
        f"止损{_fmt(rr_info.get('stop_price'), 2)}元 / 止盈{_fmt(rr_info.get('target_price'), 2)}元",
    ]

    pe = fund.get("pe_ttm")
    pb = fund.get("pb")
    pe_s = _fmt(pe, 1)
    pb_s = _fmt(pb, 2)
    if pe_s != "暂缺" or pb_s != "暂缺":
        parts.append(f"估值 PE={pe_s} / PB={pb_s}")

    rev = _fmt(fund.get("revenue_yoy"), 1, "%")
    profit = _fmt(fund.get("profit_yoy"), 1, "%")
    if rev != "暂缺" or profit != "暂缺":
        parts.append(f"营收同比{rev}、净利润同比{profit}")
    else:
        parts.append("基本面增速字段暂缺，建议人工复核财报")

    parts.append("适合约1-2个月持有观察，不追涨、少看盘")
    return "；".join(parts) + "。"
