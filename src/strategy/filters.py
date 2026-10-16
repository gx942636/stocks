"""中线低位企稳。只使用买入日及更早的行情。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def is_main_board_code(code: str, strategy: dict[str, Any]) -> bool:
    code = str(code).zfill(6)
    if strategy.get("exclude_kcb", True) and code.startswith("688"):
        return False
    if strategy.get("exclude_cyb", True) and (code.startswith("300") or code.startswith("301")):
        return False
    # 主板：沪市 60，深市 000/001/002/003；排除北交所 8/4 开头
    if code.startswith(("8", "4", "9")):
        return False
    return code.startswith(("60", "000", "001", "002", "003"))


def pass_name_filters(name: str, is_st: bool, strategy: dict[str, Any]) -> bool:
    name = str(name or "")
    if strategy.get("exclude_st", True):
        if is_st or "ST" in name.upper() or "退" in name:
            return False
    return True


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def enrich_daily(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    """为日线增加 60 日回撤、20 日低点和近 5 日是否再创新低。"""
    if df is None or df.empty:
        return df
    out = df.sort_values("date").copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    if "pct_chg" not in out.columns:
        out["pct_chg"] = np.nan
    out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")
    if out["pct_chg"].isna().all():
        out["pct_chg"] = out["close"].pct_change() * 100

    pullback_days = int(strategy.get("pullback_days", 60))
    low_days = int(strategy.get("low_days", 20))
    stable_days = int(strategy.get("stable_days", 5))
    amount_window = int(strategy.get("amount_window", 60))
    base_days = low_days - stable_days

    high_n = out["high"].rolling(pullback_days, min_periods=pullback_days).max()
    out["high_60"] = high_n
    out["drawdown_60"] = (high_n - out["close"]) / high_n.replace(0, np.nan) * 100
    out["low_20"] = out["low"].rolling(low_days, min_periods=low_days).min()
    out["low_dist_pct"] = (out["close"] - out["low_20"]) / out["low_20"].replace(0, np.nan) * 100
    # 20 日窗口里、最近 5 日之前的低点。最近 5 日再跌破它，就还没止跌。
    if base_days <= 0:
        out["base_low"] = np.nan
    else:
        out["base_low"] = out["low"].shift(stable_days).rolling(base_days, min_periods=base_days).min()
    out["recent_low_5"] = out["low"].rolling(stable_days, min_periods=stable_days).min()
    out["ret_5d"] = out["close"].pct_change(stable_days) * 100

    amount = pd.to_numeric(out["amount"], errors="coerce") if "amount" in out.columns else np.nan
    out["amount_median"] = pd.Series(amount, index=out.index).rolling(
        amount_window, min_periods=amount_window
    ).median()
    return out


def evaluate_bar(
    row: pd.Series,
    strategy: dict[str, Any],
    circ_mv: float | None = None,
    fund: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, float]]:
    """判断买入日是否符合中线低位企稳。返回 (是否通过, 指标字典)。"""
    fund = fund or {}
    mv = _num(circ_mv)
    if np.isnan(mv):
        mv = _num(fund.get("circ_mv"))
    metrics = {
        "close": _num(row.get("close")),
        "pct_chg": _num(row.get("pct_chg")),
        "drawdown_60": _num(row.get("drawdown_60")),
        "low_20": _num(row.get("low_20")),
        "low_dist_pct": _num(row.get("low_dist_pct")),
        "base_low": _num(row.get("base_low")),
        "recent_low_5": _num(row.get("recent_low_5")),
        "ret_5d": _num(row.get("ret_5d")),
        "amount_median": _num(row.get("amount_median")),
        "high_60": _num(row.get("high_60")),
        "pe_ttm": _num(fund.get("pe_ttm")),
        "pb": _num(fund.get("pb")),
        "circ_mv": mv,
    }

    price = metrics["close"]
    if np.isnan(price) or price < float(strategy.get("price_min", 2)):
        return False, metrics
    if np.isnan(mv) or mv < float(strategy.get("circ_mv_min_yi", 30)):
        return False, metrics

    pe = metrics["pe_ttm"]
    pb = metrics["pb"]
    if np.isnan(pe) or not (0 < pe <= float(strategy.get("pe_max", 80))):
        return False, metrics
    if np.isnan(pb) or not (0 < pb <= float(strategy.get("pb_max", 8))):
        return False, metrics

    drawdown = metrics["drawdown_60"]
    dd_min = float(strategy.get("drawdown_min", 15))
    dd_max = float(strategy.get("drawdown_max", 30))
    if np.isnan(drawdown) or not (dd_min <= drawdown <= dd_max):
        return False, metrics

    low_20 = metrics["low_20"]
    base_low = metrics["base_low"]
    recent_low = metrics["recent_low_5"]
    dist = metrics["low_dist_pct"]
    if np.isnan(low_20) or low_20 <= 0 or np.isnan(base_low) or np.isnan(recent_low) or np.isnan(dist):
        return False, metrics
    if recent_low < base_low:
        return False, metrics
    if dist < 0 or dist > float(strategy.get("low_dist_max", 6)):
        return False, metrics

    ret_5d = metrics["ret_5d"]
    if np.isnan(ret_5d) or ret_5d > float(strategy.get("ret_5d_max", 6)):
        return False, metrics
    pct = metrics["pct_chg"]
    if np.isnan(pct) or not (
        float(strategy.get("pct_chg_min", -4)) <= pct <= float(strategy.get("pct_chg_max", 3))
    ):
        return False, metrics

    amount_min = float(strategy.get("amount_median_min_yi", 0.8)) * 1e8
    if np.isnan(metrics["amount_median"]) or metrics["amount_median"] < amount_min:
        return False, metrics
    return True, metrics
