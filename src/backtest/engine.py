"""回测撮合引擎。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Position:
    code: str
    name: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    hold_max_days: int
    risk_per_share: float
    bars_held: int = 0


@dataclass
class Trade:
    code: str
    name: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    ret_pct: float
    pnl: float
    reason: str
    realized_rr: float


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_cash: float = 0.0
    final_equity: float = 0.0


def run_backtest(
    signals: pd.DataFrame,
    daily_map: dict[str, pd.DataFrame],
    strategy: dict[str, Any],
    backtest_cfg: dict[str, Any],
    name_map: dict[str, str] | None = None,
) -> BacktestResult:
    """
    signals 列: date, code, stop_price, target_price, signal_close
    daily_map: code -> 全日线（需含 open/high/low/close/date）
    规则: 信号日收盘后入选，次日开盘买入；止损/止盈用当日高低价触碰；到期收盘卖。
    """
    name_map = name_map or {}
    initial_cash = float(backtest_cfg.get("initial_cash", 1_000_000))
    max_positions = int(strategy.get("max_positions", 5))
    hold_max = int(strategy.get("hold_max_days", 40))

    if signals is None or signals.empty:
        return BacktestResult(final_cash=initial_cash, final_equity=initial_cash)

    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

    # 交易日历：合并所有股票日期
    all_dates = sorted({d for df in daily_map.values() for d in df["date"].tolist()})
    if not all_dates:
        return BacktestResult(final_cash=initial_cash, final_equity=initial_cash)

    cash = initial_cash
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity_rows: list[dict[str, Any]] = []

    # 待买入：信号日 -> list
    pending: dict[pd.Timestamp, list[pd.Series]] = {}
    for _, row in signals.iterrows():
        pending.setdefault(pd.Timestamp(row["date"]), []).append(row)

    for i, dt in enumerate(all_dates):
        # 1) 先处理卖出（开盘后用当日高低）
        to_close: list[tuple[str, float, str]] = []
        for code, pos in list(positions.items()):
            df = daily_map.get(code)
            if df is None:
                continue
            bar = df[df["date"] == dt]
            if bar.empty:
                pos.bars_held += 1
                continue
            bar = bar.iloc[0]
            pos.bars_held += 1
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            reason = None
            exit_price = close
            if low <= pos.stop_price:
                reason = "止损"
                exit_price = min(float(bar["open"]), pos.stop_price)
            elif high >= pos.target_price:
                reason = "止盈"
                exit_price = max(float(bar["open"]), pos.target_price)
            elif pos.bars_held >= pos.hold_max_days:
                reason = "到期"
                exit_price = close

            if reason:
                to_close.append((code, exit_price, reason))

        for code, exit_price, reason in to_close:
            pos = positions.pop(code)
            proceeds = exit_price * pos.shares
            cash += proceeds
            pnl = (exit_price - pos.entry_price) * pos.shares
            ret_pct = (exit_price / pos.entry_price - 1.0) * 100
            realized_rr = (exit_price - pos.entry_price) / pos.risk_per_share if pos.risk_per_share > 0 else 0.0
            trades.append(
                Trade(
                    code=code,
                    name=pos.name,
                    entry_date=str(pos.entry_date.date()),
                    exit_date=str(pd.Timestamp(dt).date()),
                    entry_price=round(pos.entry_price, 4),
                    exit_price=round(exit_price, 4),
                    shares=pos.shares,
                    ret_pct=round(ret_pct, 4),
                    pnl=round(pnl, 2),
                    reason=reason,
                    realized_rr=round(realized_rr, 4),
                )
            )

        # 2) 执行昨日信号的次日开盘买入
        if i > 0:
            prev_dt = all_dates[i - 1]
            for sig in pending.get(pd.Timestamp(prev_dt), []):
                code = str(sig["code"]).zfill(6)
                if code in positions:
                    continue
                if len(positions) >= max_positions:
                    break
                df = daily_map.get(code)
                if df is None:
                    continue
                bar = df[df["date"] == dt]
                if bar.empty:
                    continue
                open_price = float(bar.iloc[0]["open"])
                if open_price <= 0 or pd.isna(open_price):
                    continue
                # 等权：剩余仓位槽位均分现金
                slots = max_positions - len(positions)
                budget = cash / slots if slots > 0 else 0
                shares = int(budget // (open_price * 100)) * 100
                if shares < 100:
                    continue
                cost = shares * open_price
                if cost > cash:
                    continue
                cash -= cost
                stop_price = float(sig["stop_price"])
                target_price = float(sig["target_price"])
                risk = open_price - stop_price
                if risk <= 0:
                    risk = open_price * float(strategy.get("stop_loss_pct", 0.12))
                positions[code] = Position(
                    code=code,
                    name=name_map.get(code, code),
                    entry_date=pd.Timestamp(dt),
                    entry_price=open_price,
                    shares=shares,
                    stop_price=stop_price,
                    target_price=target_price,
                    hold_max_days=hold_max,
                    risk_per_share=risk,
                    bars_held=0,
                )

        # 3) 权益
        mtm = cash
        for code, pos in positions.items():
            df = daily_map.get(code)
            px = pos.entry_price
            if df is not None:
                bar = df[df["date"] == dt]
                if not bar.empty:
                    px = float(bar.iloc[0]["close"])
            mtm += px * pos.shares
        equity_rows.append({"date": dt, "equity": mtm, "cash": cash, "positions": len(positions)})

    # 期末强平
    if positions and all_dates:
        last_dt = all_dates[-1]
        for code, pos in list(positions.items()):
            df = daily_map.get(code)
            exit_price = pos.entry_price
            if df is not None:
                bar = df[df["date"] == last_dt]
                if not bar.empty:
                    exit_price = float(bar.iloc[0]["close"])
            cash += exit_price * pos.shares
            pnl = (exit_price - pos.entry_price) * pos.shares
            ret_pct = (exit_price / pos.entry_price - 1.0) * 100
            realized_rr = (exit_price - pos.entry_price) / pos.risk_per_share if pos.risk_per_share > 0 else 0.0
            trades.append(
                Trade(
                    code=code,
                    name=pos.name,
                    entry_date=str(pos.entry_date.date()),
                    exit_date=str(pd.Timestamp(last_dt).date()),
                    entry_price=round(pos.entry_price, 4),
                    exit_price=round(exit_price, 4),
                    shares=pos.shares,
                    ret_pct=round(ret_pct, 4),
                    pnl=round(pnl, 2),
                    reason="期末强平",
                    realized_rr=round(realized_rr, 4),
                )
            )
        positions.clear()

    equity_curve = pd.DataFrame(equity_rows)
    final_equity = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else cash
    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        final_cash=cash,
        final_equity=final_equity,
    )
