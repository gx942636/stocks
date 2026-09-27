"""回测绩效指标。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .engine import BacktestResult, Trade


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "code",
                "name",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "shares",
                "ret_pct",
                "pnl",
                "reason",
                "realized_rr",
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades])


def summarize_trades(result: BacktestResult, initial_cash: float) -> dict[str, Any]:
    trades = result.trades
    n = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_realized_rr": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "final_equity": result.final_equity,
        }

    rets = np.array([t.ret_pct for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    rr = np.array([t.realized_rr for t in trades], dtype=float)
    total_pnl = float(sum(t.pnl for t in trades))

    max_dd = 0.0
    if result.equity_curve is not None and not result.equity_curve.empty:
        eq = result.equity_curve["equity"].astype(float)
        peak = eq.cummax()
        dd = (eq - peak) / peak.replace(0, np.nan)
        max_dd = float(dd.min() * 100) if len(dd) else 0.0

    total_return = (result.final_equity / initial_cash - 1.0) * 100 if initial_cash else 0.0

    return {
        "trades": n,
        "win_rate": round(float(len(wins) / n * 100), 2),
        "avg_win_pct": round(float(wins.mean()) if len(wins) else 0.0, 4),
        "avg_loss_pct": round(float(losses.mean()) if len(losses) else 0.0, 4),
        "avg_realized_rr": round(float(np.nanmean(rr)) if len(rr) else 0.0, 4),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "final_equity": round(result.final_equity, 2),
    }
