"""取数 → 筛选 → 回测 → 导出。"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.backtest.engine import run_backtest
from src.backtest.metrics import summarize_trades, trades_to_frame
from src.data.base import DataProvider
from src.strategy.filters import (
    enrich_daily,
    evaluate_bar,
    is_main_board_code,
    pass_name_filters,
)
from src.strategy.reasons import build_reason
from src.strategy.risk_reward import compute_risk_reward


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
    fetch_start = _lookback_start(start_date, int(strategy.get("history_calendar_days", 400)))
    min_bars = int(strategy.get("lookback_days", 250))
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
        daily = enrich_daily(daily, strategy)
        daily_map[code] = daily
        name_map[code] = name
        circ = row.get("circ_mv")
        circ_mv_map[code] = float(circ) if pd.notna(circ) else float("nan")

    return daily_map, name_map, circ_mv_map


def generate_signals(
    daily_map: dict[str, pd.DataFrame],
    circ_mv_map: dict[str, float],
    name_map: dict[str, str],
    start_date: str,
    end_date: str,
    strategy: dict[str, Any],
    fund_map: dict[str, dict[str, Any]] | None = None,
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """区间内逐日扫描。盈亏比只留给回测参考，不作为入选条件。"""
    fund_map = fund_map or {}
    rows: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    items = list(daily_map.items())
    for code, daily in _iter_with_progress(items, "扫描信号", progress):
        fund = fund_map.get(code, {"name": name_map.get(code, code)})
        sub = daily[(daily["date"] >= start_ts) & (daily["date"] <= end_ts)]
        for _, bar in sub.iterrows():
            ok, metrics = evaluate_bar(bar, strategy, circ_mv_map.get(code), fund)
            if not ok:
                continue
            rr_info = compute_risk_reward(metrics["close"], metrics["high_lookback"], strategy)
            reason = build_reason(metrics, rr_info, fund)
            rows.append(
                {
                    "date": bar["date"],
                    "code": code,
                    "name": name_map.get(code, code),
                    "signal_close": metrics["close"],
                    "pct_chg": metrics["pct_chg"],
                    "drawdown_250": metrics["drawdown_250"],
                    "range_pos": metrics["range_pos"],
                    "amount_median": metrics["amount_median"],
                    "high_lookback": metrics["high_lookback"],
                    "low_lookback": metrics["low_lookback"],
                    "stop_price": rr_info["stop_price"],
                    "target_price": rr_info["target_price"],
                    "rr": rr_info["rr"],
                    "hold_days": rr_info["hold_days"],
                    "pe_ttm": metrics["pe_ttm"],
                    "pb": metrics["pb"],
                    "circ_mv": metrics["circ_mv"],
                    "reason": reason,
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["date", "range_pos", "code"], ascending=[True, True, True])
    return out.reset_index(drop=True)


def screen_period(signals: pd.DataFrame) -> pd.DataFrame:
    """区间内按股票去重，保留最后一次符合的日期，并给出符合天数。"""
    if signals is None or signals.empty:
        return pd.DataFrame()
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    counts = sig.groupby("code").size().rename("hit_days").reset_index()
    latest = sig.loc[sig.groupby("code")["date"].idxmax()].copy()
    latest = latest.merge(counts, on="code", how="left")
    latest = latest.sort_values(["range_pos", "drawdown_250"], ascending=[True, False])
    return latest.reset_index(drop=True)


def first_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """每只股票只保留区间内第一次符合，避免回测里每天重复买入。"""
    if signals is None or signals.empty:
        return pd.DataFrame()
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig.sort_values(["date", "range_pos", "code"], ascending=[True, True, True])
    return sig.drop_duplicates("code", keep="first").reset_index(drop=True)


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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    xlsx = output_dir / f"midterm_ambush_{stamp}.xlsx"

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
    start_date = str(config["start_date"])
    end_date = str(config["end_date"])
    mode = str(config.get("mode", "both")).lower()
    strategy = config.get("strategy", {})
    backtest_cfg = config.get("backtest", {})

    universe = provider.list_universe()
    print(f"股票池数量: {len(universe)}（数据源={config.get('data', {}).get('source')}）")

    daily_map, name_map, circ_mv_map = collect_stock_data(
        provider, universe, start_date, end_date, strategy
    )
    print(f"有效日线股票数: {len(daily_map)}")

    fund_map = build_fund_map(provider, list(daily_map.keys()))
    # 若 universe 有流通市值则优先，否则用基本面
    for code, f in fund_map.items():
        if code in circ_mv_map and (pd.isna(circ_mv_map[code]) or circ_mv_map[code] != circ_mv_map[code]):
            if pd.notna(f.get("circ_mv")):
                circ_mv_map[code] = float(f["circ_mv"])

    signals = generate_signals(
        daily_map, circ_mv_map, name_map, start_date, end_date, strategy, fund_map
    )
    print(f"区间信号条数: {0 if signals is None or signals.empty else len(signals)}")

    candidates = screen_period(signals)
    print(f"区间符合标的: {0 if candidates is None or candidates.empty else len(candidates)}")

    summary: dict[str, Any] = {}
    trades_df = pd.DataFrame()
    equity = pd.DataFrame()

    if mode in ("backtest", "both") and signals is not None and not signals.empty:
        bt_signals = first_signals(signals)
        result = run_backtest(bt_signals, daily_map, strategy, backtest_cfg, name_map)
        summary = summarize_trades(result, float(backtest_cfg.get("initial_cash", 1_000_000)))
        trades_df = trades_to_frame(result.trades)
        equity = result.equity_curve
        print("回测摘要:", summary)
    elif mode in ("backtest", "both"):
        print("无信号，跳过回测。")

    output_dir = project_root / "output"
    xlsx = export_results(output_dir, candidates, signals, summary, trades_df, equity)
    print(f"结果已写入: {xlsx}")

    return {
        "candidates": candidates,
        "signals": signals,
        "summary": summary,
        "trades": trades_df,
        "equity": equity,
        "output": xlsx,
    }
