"""中线埋伏技术过滤。"""
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


def _limit_up_flag(pct_chg: float, threshold: float) -> bool:
    return (not np.isnan(pct_chg)) and pct_chg >= threshold


def enrich_daily(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    """为日线增加策略所需衍生列。"""
    if df is None or df.empty:
        return df
    out = df.sort_values("date").copy()
    out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")
    if out["pct_chg"].isna().all():
        out["pct_chg"] = out["close"].pct_change() * 100

    # 量比近似：当日量 / 过去5日均量
    vol_ma5 = out["volume"].rolling(5, min_periods=5).mean().shift(1)
    out["volume_ratio"] = out["volume"] / vol_ma5.replace(0, np.nan)

    out["ret_5d"] = out["close"].pct_change(5) * 100
    out["ret_60d"] = out["close"].pct_change(60) * 100
    high_60 = out["high"].rolling(60, min_periods=60).max()
    out["high_60"] = high_60
    out["drawdown_60d"] = (high_60 - out["close"]) / high_60.replace(0, np.nan) * 100

    thr = float(strategy.get("limit_up_threshold", 9.5))
    out["is_limit_up"] = out["pct_chg"].map(lambda x: _limit_up_flag(float(x) if pd.notna(x) else np.nan, thr))
    out["had_limit_up_20"] = out["is_limit_up"].rolling(20, min_periods=1).max().astype(bool)

    # 连涨天数（含当日）
    up = (out["pct_chg"] > 0).astype(int)
    streak = []
    cur = 0
    for v in up.tolist():
        cur = cur + 1 if v == 1 else 0
        streak.append(cur)
    out["consecutive_up"] = streak
    return out


def evaluate_bar(
    row: pd.Series,
    strategy: dict[str, Any],
    circ_mv: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """判断单日是否满足埋伏条件。返回 (是否通过, 指标字典)。"""
    metrics = {
        "pct_chg": float(row.get("pct_chg", np.nan)),
        "turnover": float(row.get("turnover", np.nan)),
        "volume_ratio": float(row.get("volume_ratio", np.nan)),
        "ret_5d": float(row.get("ret_5d", np.nan)),
        "ret_60d": float(row.get("ret_60d", np.nan)),
        "drawdown_60d": float(row.get("drawdown_60d", np.nan)),
        "consecutive_up": float(row.get("consecutive_up", np.nan)),
        "close": float(row.get("close", np.nan)),
        "high_60": float(row.get("high_60", np.nan)),
    }

    def between(val: float, lo: float, hi: float) -> bool:
        return pd.notna(val) and lo <= val <= hi

    price = metrics["close"]
    if not between(price, float(strategy["price_min"]), float(strategy["price_max"])):
        return False, metrics

    if circ_mv is not None and pd.notna(circ_mv):
        if not between(float(circ_mv), float(strategy["circ_mv_min_yi"]), float(strategy["circ_mv_max_yi"])):
            return False, metrics

    if not between(metrics["pct_chg"], float(strategy["pct_chg_min"]), float(strategy["pct_chg_max"])):
        return False, metrics
    if metrics["pct_chg"] < float(strategy["daily_drop_floor"]):
        return False, metrics
    if not between(metrics["turnover"], float(strategy["turnover_min"]), float(strategy["turnover_max"])):
        return False, metrics
    if not between(
        metrics["volume_ratio"],
        float(strategy["volume_ratio_min"]),
        float(strategy["volume_ratio_max"]),
    ):
        return False, metrics
    if bool(row.get("had_limit_up_20", False)):
        return False, metrics
    if metrics["consecutive_up"] > float(strategy["consecutive_up_max"]):
        return False, metrics
    if not between(metrics["ret_60d"], float(strategy["ret_60d_min"]), float(strategy["ret_60d_max"])):
        return False, metrics
    if not between(
        metrics["drawdown_60d"],
        float(strategy["drawdown_60d_min"]),
        float(strategy["drawdown_60d_max"]),
    ):
        return False, metrics
    if not between(metrics["ret_5d"], float(strategy["ret_5d_min"]), float(strategy["ret_5d_max"])):
        return False, metrics

    return True, metrics
