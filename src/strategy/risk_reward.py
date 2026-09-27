"""盈亏比计算。"""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_risk_reward(
    buy_price: float,
    high_60: float,
    strategy: dict[str, Any],
) -> dict[str, float]:
    """
    止损：buy * (1 - stop_loss_pct)
    目标：buy + (high_60 - buy) * repair_ratio，并受 take_profit_cap_pct 上限约束
    R = 收益空间 / 风险空间
    """
    if buy_price is None or np.isnan(buy_price) or buy_price <= 0:
        return {
            "buy_price": float("nan"),
            "stop_price": float("nan"),
            "target_price": float("nan"),
            "reward": float("nan"),
            "risk": float("nan"),
            "rr": float("nan"),
            "hold_days": float(strategy.get("hold_max_days", 40)),
        }

    stop_pct = float(strategy.get("stop_loss_pct", 0.12))
    cap_pct = float(strategy.get("take_profit_cap_pct", 0.25))
    repair = float(strategy.get("repair_ratio", 0.60))

    stop_price = buy_price * (1.0 - stop_pct)
    risk = buy_price - stop_price

    if high_60 is None or np.isnan(high_60) or high_60 <= buy_price:
        repair_target = buy_price * (1.0 + min(cap_pct, 0.18))
    else:
        repair_target = buy_price + (high_60 - buy_price) * repair

    cap_target = buy_price * (1.0 + cap_pct)
    target_price = min(repair_target, cap_target)
    # 目标至少高于买入，否则 R 无意义
    if target_price <= buy_price:
        target_price = buy_price * (1.0 + 0.18)

    reward = target_price - buy_price
    rr = reward / risk if risk > 0 else float("nan")

    hold_days = float(
        (float(strategy.get("hold_min_days", 20)) + float(strategy.get("hold_max_days", 40))) / 2.0
    )

    return {
        "buy_price": round(buy_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "reward": round(reward, 4),
        "risk": round(risk, 4),
        "rr": round(float(rr), 4) if rr == rr else float("nan"),
        "hold_days": hold_days,
    }
