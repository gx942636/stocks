"""中线低位筛选。"""
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
    """为日线增加一年高低点、区间位置和成交额中位数。"""
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

    lookback = int(strategy.get("lookback_days", 250))
    amount_window = int(strategy.get("amount_window", 120))
    high_n = out["high"].rolling(lookback, min_periods=lookback).max()
    low_n = out["low"].rolling(lookback, min_periods=lookback).min()
    span = (high_n - low_n).replace(0, np.nan)
    out["high_lookback"] = high_n
    out["low_lookback"] = low_n
    out["drawdown_250"] = (high_n - out["close"]) / high_n.replace(0, np.nan) * 100
    out["range_pos"] = (out["close"] - low_n) / span

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
    """判断单日是否满足低位潜伏。返回 (是否通过, 指标字典)。"""
    fund = fund or {}
    mv = _num(circ_mv)
    if np.isnan(mv):
        mv = _num(fund.get("circ_mv"))
    metrics = {
        "close": _num(row.get("close")),
        "pct_chg": _num(row.get("pct_chg")),
        "drawdown_250": _num(row.get("drawdown_250")),
        "range_pos": _num(row.get("range_pos")),
        "amount_median": _num(row.get("amount_median")),
        "high_lookback": _num(row.get("high_lookback")),
        "low_lookback": _num(row.get("low_lookback")),
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
    pe_max = float(strategy.get("pe_max", 150))
    pb_max = float(strategy.get("pb_max", 15))
    if np.isnan(pe) or not (0 < pe <= pe_max):
        return False, metrics
    if np.isnan(pb) or not (0 < pb <= pb_max):
        return False, metrics

    drawdown = metrics["drawdown_250"]
    if np.isnan(drawdown) or drawdown < float(strategy.get("drawdown_min", 30)):
        return False, metrics
    pos = metrics["range_pos"]
    if np.isnan(pos) or pos < 0 or pos > float(strategy.get("range_pos_max", 0.35)):
        return False, metrics

    amount_min = float(strategy.get("amount_median_min_yi", 0.5)) * 1e8
    if np.isnan(metrics["amount_median"]) or metrics["amount_median"] < amount_min:
        return False, metrics

    return True, metrics
