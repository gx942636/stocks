"""短线两套独立方案：趋势涨、回踩止跌。

收盘后筛选，供次日早盘买入。只使用信号日及更早的行情。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    number = _num(value)
    if number != number:
        return "暂缺"
    return f"{number:.{digits}f}{suffix}"


def enrich_short(df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
    """补上均线、20 日涨幅、10 日回撤、8 日低点和量比。"""
    if df is None or df.empty:
        return df
    out = df.sort_values("date").copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    span = out["high"] - out["low"]
    out["close_pos"] = (out["close"] - out["low"]) / span.replace(0, np.nan)
    out["amplitude"] = span / out["close"].replace(0, np.nan) * 100
    if "pct_chg" not in out.columns:
        out["pct_chg"] = np.nan
    out["pct_chg"] = pd.to_numeric(out["pct_chg"], errors="coerce")
    if out["pct_chg"].isna().all():
        out["pct_chg"] = out["close"].pct_change() * 100

    fast = int(strategy.get("ma_fast", 10))
    slow = int(strategy.get("ma_slow", 20))
    high_days = int(strategy.get("high_days", 10))
    low_days = int(strategy.get("pull_low_days", 8))
    stable_days = int(strategy.get("pull_stable_days", 3))
    base_days = int(strategy.get("pull_base_days", 5))
    amount_window = int(strategy.get("amount_window", 20))

    out["ma_fast"] = out["close"].rolling(fast, min_periods=fast).mean()
    out["ma_slow"] = out["close"].rolling(slow, min_periods=slow).mean()
    out["ret_20d"] = out["close"].pct_change(slow) * 100
    out["ret_3d"] = out["close"].pct_change(stable_days) * 100

    high_n = out["high"].rolling(high_days, min_periods=high_days).max()
    out["high_10"] = high_n
    out["drawdown_10"] = (high_n - out["close"]) / high_n.replace(0, np.nan) * 100

    out["low_8"] = out["low"].rolling(low_days, min_periods=low_days).min()
    out["low_dist_8"] = (out["close"] - out["low_8"]) / out["low_8"].replace(0, np.nan) * 100
    out["recent_low"] = out["low"].rolling(stable_days, min_periods=stable_days).min()
    if base_days <= 0:
        out["base_low"] = np.nan
    else:
        out["base_low"] = out["low"].shift(stable_days).rolling(base_days, min_periods=base_days).min()

    amount = pd.to_numeric(out["amount"], errors="coerce") if "amount" in out.columns else np.nan
    amount = pd.Series(amount, index=out.index)
    out["amount_median"] = amount.rolling(amount_window, min_periods=amount_window).median()
    out["amount_ratio"] = amount / out["amount_median"].replace(0, np.nan)

    resist_price, resist_room, resist_state = _resistance_columns(
        out["high"],
        out["close"],
        gap=int(strategy.get("resist_gap_days", 10)),
        lookback=int(strategy.get("resist_lookback", 60)),
        min_prior=int(strategy.get("resist_min_prior_bars", 20)),
    )
    out["resist_price"] = resist_price
    out["resist_room"] = resist_room
    out["resist_state"] = resist_state
    return out


def _resistance_columns(
    high: pd.Series,
    close: pd.Series,
    gap: int,
    lookback: int,
    min_prior: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最近上方压力：去掉最近 gap 日之后，回看窗口内仍高于收盘的最低高点。"""
    highs = high.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    n = len(highs)
    price = np.full(n, np.nan)
    room = np.full(n, np.nan)
    state = np.zeros(n, dtype=float)
    gap = max(int(gap), 0)
    lookback = max(int(lookback), 1)
    min_prior = max(int(min_prior), 1)
    for i in range(n):
        end = i - gap + 1
        if end <= 0:
            continue
        start = max(0, end - lookback)
        window = highs[start:end]
        window = window[np.isfinite(window)]
        if window.size < min_prior:
            continue
        px = closes[i]
        if not np.isfinite(px) or px <= 0:
            continue
        above = window[window > px]
        if above.size == 0:
            state[i] = 1.0
            continue
        level = float(above.min())
        price[i] = level
        room[i] = (level - px) / px * 100.0
        state[i] = 2.0
    return price, room, state


def _metrics(row: pd.Series, circ_mv: float | None) -> dict[str, float]:
    mv = _num(circ_mv)
    return {
        "close": _num(row.get("close")),
        "pct_chg": _num(row.get("pct_chg")),
        "ma_fast": _num(row.get("ma_fast")),
        "ma_slow": _num(row.get("ma_slow")),
        "ret_20d": _num(row.get("ret_20d")),
        "ret_3d": _num(row.get("ret_3d")),
        "drawdown_10": _num(row.get("drawdown_10")),
        "low_8": _num(row.get("low_8")),
        "low_dist_8": _num(row.get("low_dist_8")),
        "recent_low": _num(row.get("recent_low")),
        "base_low": _num(row.get("base_low")),
        "amount_median": _num(row.get("amount_median")),
        "amount_ratio": _num(row.get("amount_ratio")),
        "close_pos": _num(row.get("close_pos")),
        "amplitude": _num(row.get("amplitude")),
        "resist_price": _num(row.get("resist_price")),
        "resist_room": _num(row.get("resist_room")),
        "resist_state": _num(row.get("resist_state")),
        "circ_mv": mv,
    }


def _pass_common(metrics: dict[str, float], strategy: dict[str, Any]) -> bool:
    price = metrics["close"]
    if np.isnan(price) or price < float(strategy.get("price_min", 2)):
        return False
    mv = metrics["circ_mv"]
    if np.isnan(mv) or mv < float(strategy.get("circ_mv_min_yi", 30)):
        return False
    amount_min = float(strategy.get("amount_median_min_yi", 1)) * 1e8
    if np.isnan(metrics["amount_median"]) or metrics["amount_median"] < amount_min:
        return False
    return True


def _pass_certainty(metrics: dict[str, float], strategy: dict[str, Any]) -> bool:
    """收盘要在全天偏上，且当天振幅不能太大，否则次日早盘方向不确定。"""
    if not _between(metrics["close_pos"], float(strategy.get("close_pos_min", 0.65)), 1.0):
        return False
    amplitude = metrics["amplitude"]
    if np.isnan(amplitude) or amplitude > float(strategy.get("amplitude_max", 6)):
        return False
    return True


def _pass_resistance(metrics: dict[str, float], strategy: dict[str, Any]) -> bool:
    """贴着更早的压力、上方空间不足，次日早盘没有上涨余地。"""
    state = metrics["resist_state"]
    if state == 1:
        return True
    if state != 2:
        return False
    room = metrics["resist_room"]
    return room == room and room >= float(strategy.get("resist_room_min", 3))


def _between(value: float, low: float, high: float) -> bool:
    return value == value and low <= value <= high


def evaluate_trend(
    row: pd.Series,
    strategy: dict[str, Any],
    circ_mv: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """趋势涨：还在涨，但当天没有涨完，收盘靠上，适合次日早盘买。"""
    metrics = _metrics(row, circ_mv)
    if (
        not _pass_common(metrics, strategy)
        or not _pass_certainty(metrics, strategy)
        or not _pass_resistance(metrics, strategy)
    ):
        return False, metrics

    close = metrics["close"]
    ma_fast = metrics["ma_fast"]
    ma_slow = metrics["ma_slow"]
    if np.isnan(ma_fast) or np.isnan(ma_slow) or not (close > ma_fast > ma_slow):
        return False, metrics

    if not _between(
        metrics["ret_20d"],
        float(strategy.get("trend_ret_20_min", 8)),
        float(strategy.get("trend_ret_20_max", 25)),
    ):
        return False, metrics
    if not _between(
        metrics["ret_3d"],
        float(strategy.get("trend_ret_3d_min", 1)),
        float(strategy.get("trend_ret_3d_max", 6)),
    ):
        return False, metrics
    if not _between(metrics["drawdown_10"], 0, float(strategy.get("trend_drawdown_max", 3))):
        return False, metrics
    if not _between(
        metrics["pct_chg"],
        float(strategy.get("trend_pct_min", 1)),
        float(strategy.get("trend_pct_max", 4)),
    ):
        return False, metrics
    if not _between(
        metrics["amount_ratio"],
        float(strategy.get("trend_amount_ratio_min", 1.2)),
        float(strategy.get("trend_amount_ratio_max", 2.2)),
    ):
        return False, metrics
    return True, metrics


def evaluate_pullback(
    row: pd.Series,
    strategy: dict[str, Any],
    circ_mv: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """回踩止跌：趋势没坏，缩量止跌已经收在全天偏上，适合次日早盘买。"""
    metrics = _metrics(row, circ_mv)
    if (
        not _pass_common(metrics, strategy)
        or not _pass_certainty(metrics, strategy)
        or not _pass_resistance(metrics, strategy)
    ):
        return False, metrics

    ret_20 = metrics["ret_20d"]
    if np.isnan(ret_20) or ret_20 < float(strategy.get("pull_ret_20_min", 8)):
        return False, metrics
    close = metrics["close"]
    ma_slow = metrics["ma_slow"]
    if np.isnan(close) or np.isnan(ma_slow) or close <= ma_slow:
        return False, metrics
    if not _between(
        metrics["drawdown_10"],
        float(strategy.get("pull_drawdown_min", 5)),
        float(strategy.get("pull_drawdown_max", 10)),
    ):
        return False, metrics

    recent_low = metrics["recent_low"]
    base_low = metrics["base_low"]
    if np.isnan(recent_low) or np.isnan(base_low) or recent_low < base_low:
        return False, metrics
    if not _between(metrics["low_dist_8"], 0, float(strategy.get("pull_low_dist_max", 3))):
        return False, metrics

    if not _between(
        metrics["ret_3d"],
        float(strategy.get("pull_ret_3d_min", -3)),
        float(strategy.get("pull_ret_3d_max", 3)),
    ):
        return False, metrics
    if not _between(
        metrics["pct_chg"],
        float(strategy.get("pull_pct_min", 0)),
        float(strategy.get("pull_pct_max", 2.5)),
    ):
        return False, metrics

    if not _between(
        metrics["amount_ratio"],
        float(strategy.get("pull_amount_ratio_min", 0.5)),
        float(strategy.get("pull_amount_ratio_max", 0.85)),
    ):
        return False, metrics
    return True, metrics


def _resist_text(metrics: dict[str, Any]) -> str:
    state = _num(metrics.get("resist_state"))
    if state == 1:
        return "已突破前高"
    if state == 2:
        return (
            f"压力{_fmt(metrics.get('resist_price'), 2)}元，"
            f"上方空间{_fmt(metrics.get('resist_room'))}%"
        )
    return "前高历史不足"


def _pos_text(metrics: dict[str, Any]) -> str:
    pos = _num(metrics.get("close_pos"))
    if pos != pos:
        return "暂缺"
    return _fmt(pos * 100, 0, "%")


def build_reason_trend(metrics: dict[str, Any]) -> str:
    return (
        f"供次日早盘买。近20日上涨{_fmt(metrics.get('ret_20d'))}%，"
        f"近3日上涨{_fmt(metrics.get('ret_3d'))}%，"
        f"当日涨幅{_fmt(metrics.get('pct_chg'))}%，"
        f"收盘位置{_pos_text(metrics)}，量比{_fmt(metrics.get('amount_ratio'), 2)}。"
        f"{_resist_text(metrics)}。"
    )


def build_reason_pullback(metrics: dict[str, Any]) -> str:
    return (
        f"供次日早盘买。近20日上涨{_fmt(metrics.get('ret_20d'))}%，"
        f"相对10日高点回撤{_fmt(metrics.get('drawdown_10'))}%，"
        f"收在8日低点上方{_fmt(metrics.get('low_dist_8'))}%，"
        f"收盘位置{_pos_text(metrics)}，当日缩量。"
        f"{_resist_text(metrics)}。"
    )
