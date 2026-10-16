"""取数 → 筛选 → 回测 → 导出。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.data.base import DataProvider
from src.strategy.filters import (
    enrich_daily,
    evaluate_bar,
    is_main_board_code,
    pass_name_filters,
)
from src.strategy.reasons import build_reason
from src.strategy.short_term import (
    build_reason_pullback,
    build_reason_trend,
    enrich_short,
    evaluate_pullback,
    evaluate_trend,
)
from src.strategy.style import active_strategy, fetch_strategy, resolve_style


ProgressFn = Callable[[int, int, str], None]


def _lookback_start(start_date: str, days: int = 120) -> str:
    dt = pd.Timestamp(start_date) - pd.Timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


def _iter_with_progress(items: list[Any], desc: str, progress: ProgressFn | None):
    """未传 progress 时走 tqdm；传入时大约每 20 条回调一次，最后一条必回调。"""
    total = len(items)
    if progress is not None and total == 0:
        progress(0, 0, desc)
    iterable: Any = items if progress is not None else tqdm(items, total=total, desc=desc)
    for i, item in enumerate(iterable, start=1):
        yield item
        if progress is not None and (i == total or i % 20 == 0):
            progress(i, total, desc)


def collect_stock_data(
    provider: DataProvider,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    strategy: dict[str, Any],
    progress: ProgressFn | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, float]]:
    """拉取并过滤股票池，返回 daily_map / name_map / circ_mv_map。"""
    fetch_start = _lookback_start(start_date, int(strategy.get("history_calendar_days", 140)))
    min_bars = int(strategy.get("pullback_days", 60))
    daily_map: dict[str, pd.DataFrame] = {}
    name_map: dict[str, str] = {}
    circ_mv_map: dict[str, float] = {}

    rows = list(universe.iterrows())
    for _, row in _iter_with_progress(rows, "拉取日线", progress):
        code = str(row["code"]).zfill(6)
        name = str(row.get("name", code))
        is_st = bool(row.get("is_st", False))
        if not is_main_board_code(code, strategy):
            continue
        if not pass_name_filters(name, is_st, strategy):
            continue

        daily = provider.get_daily(code, fetch_start, end_date)
        if daily is None or daily.empty or len(daily) < min_bars:
            continue
        daily_map[code] = daily
        name_map[code] = name
        circ = row.get("circ_mv")
        circ_mv_map[code] = float(circ) if pd.notna(circ) else float("nan")

    return daily_map, name_map, circ_mv_map


def screen_asof(
    daily_map: dict[str, pd.DataFrame],
    circ_mv_map: dict[str, float],
    name_map: dict[str, str],
    as_of: str,
    strategy: dict[str, Any],
    fund_map: dict[str, dict[str, Any]] | None = None,
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """假设在 as_of 买入。非交易日用不超过它的最近交易日，只判断这一天。"""
    fund_map = fund_map or {}
    as_of_ts = pd.Timestamp(as_of)
    stale_before = as_of_ts - pd.Timedelta(days=15)
    rows: list[dict[str, Any]] = []
    items = list(daily_map.items())
    for code, daily in _iter_with_progress(items, "筛选买入日", progress):
        hist = daily[daily["date"] <= as_of_ts]
        if hist.empty:
            continue
        hist = enrich_daily(hist, strategy)
        bar = hist.iloc[-1]
        if pd.Timestamp(bar["date"]) < stale_before:
            continue
        fund = fund_map.get(code, {"name": name_map.get(code, code)})
        ok, metrics = evaluate_bar(bar, strategy, circ_mv_map.get(code), fund)
        if not ok:
            continue
        rows.append(
            {
                "date": bar["date"],
                "code": code,
                "name": name_map.get(code, code),
                "signal_close": metrics["close"],
                "pct_chg": metrics["pct_chg"],
                "drawdown_60": metrics["drawdown_60"],
                "low_20": metrics["low_20"],
                "low_dist_pct": metrics["low_dist_pct"],
                "ret_5d": metrics["ret_5d"],
                "amount_median": metrics["amount_median"],
                "pe_ttm": metrics["pe_ttm"],
                "pb": metrics["pb"],
                "circ_mv": metrics["circ_mv"],
                "reason": build_reason(metrics),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(["low_dist_pct", "code"], ascending=[True, True])
    return out.reset_index(drop=True)


def _short_row(
    code: str,
    name: str,
    bar: pd.Series,
    metrics: dict[str, float],
    scheme: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "date": bar["date"],
        "code": code,
        "name": name,
        "scheme": scheme,
        "signal_close": metrics["close"],
        "pct_chg": metrics["pct_chg"],
        "ret_20d": metrics["ret_20d"],
        "drawdown_10": metrics["drawdown_10"],
        "low_8": metrics["low_8"],
        "low_dist_8": metrics["low_dist_8"],
        "ret_3d": metrics["ret_3d"],
        "amount_ratio": metrics["amount_ratio"],
        "close_pos": metrics["close_pos"],
        "resist_price": metrics["resist_price"],
        "resist_room": metrics["resist_room"],
        "resist_state": metrics["resist_state"],
        "amount_median": metrics["amount_median"],
        "circ_mv": metrics["circ_mv"],
        "reason": reason,
    }


def screen_short_asof(
    daily_map: dict[str, pd.DataFrame],
    circ_mv_map: dict[str, float],
    name_map: dict[str, str],
    as_of: str,
    strategy: dict[str, Any],
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """短线：趋势涨和回踩止跌各判一次。两种都符合时各留一行。"""
    as_of_ts = pd.Timestamp(as_of)
    stale_before = as_of_ts - pd.Timedelta(days=15)
    trend_rows: list[dict[str, Any]] = []
    pull_rows: list[dict[str, Any]] = []
    items = list(daily_map.items())
    for code, daily in _iter_with_progress(items, "筛选买入日", progress):
        hist = daily[daily["date"] <= as_of_ts]
        if hist.empty:
            continue
        hist = enrich_short(hist, strategy)
        bar = hist.iloc[-1]
        if pd.Timestamp(bar["date"]) < stale_before:
            continue
        name = name_map.get(code, code)
        circ = circ_mv_map.get(code)
        trend_ok, trend_metrics = evaluate_trend(bar, strategy, circ)
        if trend_ok:
            trend_rows.append(
                _short_row(code, name, bar, trend_metrics, "趋势涨", build_reason_trend(trend_metrics))
            )
        pull_ok, pull_metrics = evaluate_pullback(bar, strategy, circ)
        if pull_ok:
            pull_rows.append(
                _short_row(code, name, bar, pull_metrics, "回踩止跌", build_reason_pullback(pull_metrics))
            )
    frames: list[pd.DataFrame] = []
    if trend_rows:
        trend = pd.DataFrame(trend_rows).sort_values(["close_pos", "code"], ascending=[False, True])
        frames.append(trend)
    if pull_rows:
        pull = pd.DataFrame(pull_rows).sort_values(
            ["low_dist_8", "close_pos", "code"], ascending=[True, False, True]
        )
        frames.append(pull)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def resolve_asof(config: dict[str, Any]) -> str:
    raw = config.get("as_of_date") or config.get("end_date") or config.get("start_date")
    return str(pd.Timestamp(raw).date())


def build_fund_map(provider: DataProvider, codes: list[str]) -> dict[str, dict[str, Any]]:
    fund = provider.get_fundamentals(codes)
    out: dict[str, dict[str, Any]] = {}
    for _, row in fund.iterrows():
        out[str(row["code"]).zfill(6)] = row.to_dict()
    return out


def export_results(
    output_dir: Path,
    candidates: pd.DataFrame,
    signals: pd.DataFrame,
    summary: dict[str, Any],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    prefix: str = "midterm_ambush",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    xlsx = output_dir / f"{prefix}_{stamp}.xlsx"

    summary_df = pd.DataFrame([summary]) if summary else pd.DataFrame()
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        if candidates is not None and not candidates.empty:
            candidates.to_excel(writer, sheet_name="candidates", index=False)
        else:
            pd.DataFrame({"msg": ["无候选股"]}).to_excel(writer, sheet_name="candidates", index=False)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="backtest_summary", index=False)
        if trades is not None and not trades.empty:
            trades.to_excel(writer, sheet_name="trades", index=False)
        if equity is not None and not equity.empty:
            equity.to_excel(writer, sheet_name="equity", index=False)
        if signals is not None and not signals.empty:
            signals.to_excel(writer, sheet_name="all_signals", index=False)

    # 同步 CSV 方便打开
    if candidates is not None and not candidates.empty:
        candidates.to_csv(output_dir / f"candidates_{stamp}.csv", index=False, encoding="utf-8-sig")
    if trades is not None and not trades.empty:
        trades.to_csv(output_dir / f"trades_{stamp}.csv", index=False, encoding="utf-8-sig")
    return xlsx


def run_pipeline(config: dict[str, Any], project_root: Path, provider: DataProvider) -> dict[str, Any]:
    as_of = resolve_asof(config)
    style = resolve_style(config)
    strategy = active_strategy(config)

    universe = provider.list_universe()
    print(f"股票池数量: {len(universe)}（数据源={config.get('data', {}).get('source')}）")

    daily_map, name_map, circ_mv_map = collect_stock_data(
        provider, universe, as_of, as_of, fetch_strategy(config)
    )
    print(f"有效日线股票数: {len(daily_map)}")

    fund_map = build_fund_map(provider, list(daily_map.keys()))
    for code, fund in fund_map.items():
        if code in circ_mv_map and (pd.isna(circ_mv_map[code]) or circ_mv_map[code] != circ_mv_map[code]):
            if pd.notna(fund.get("circ_mv")):
                circ_mv_map[code] = float(fund["circ_mv"])

    if style == "short":
        candidates = screen_short_asof(
            daily_map, circ_mv_map, name_map, as_of, strategy
        )
    else:
        candidates = screen_asof(
            daily_map, circ_mv_map, name_map, as_of, strategy, fund_map
        )
    n_candidates = 0 if candidates is None or candidates.empty else len(candidates)
    print(f"假设买入日符合标的: {n_candidates}")

    output_dir = project_root / "output"
    prefix = "short_screen" if style == "short" else "midterm_ambush"
    xlsx = export_results(
        output_dir, candidates, candidates, {}, pd.DataFrame(), pd.DataFrame(), prefix=prefix
    )
    print(f"结果已写入: {xlsx}")

    return {
        "candidates": candidates,
        "as_of": as_of,
        "style": style,
        "output": xlsx,
    }
