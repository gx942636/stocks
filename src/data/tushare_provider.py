"""Tushare 数据适配（可选，需 token）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import DataProvider


def _normalize_code(code: str) -> str:
    code = str(code).strip()
    if "." in code:
        return code.split(".")[0].zfill(6)
    return code.zfill(6)


def _ts_code(code: str) -> str:
    code = _normalize_code(code)
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


class TushareProvider(DataProvider):
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        super().__init__(config, project_root)
        token = self.data_cfg.get("tushare_token") or ""
        if not token:
            raise ValueError("config.data.tushare_token 为空，请填写后再使用 source=tushare")
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("请先安装 tushare: pip install tushare") from exc
        self.pro = ts.pro_api(token)
        self._basic: pd.DataFrame | None = None

    def _load_basic(self) -> pd.DataFrame:
        if self._basic is not None:
            return self._basic
        cache_path = self.cache_dir / "tushare_basic.csv"
        if cache_path.exists():
            self._basic = pd.read_csv(cache_path, dtype={"symbol": str, "code": str})
            self._basic["code"] = self._basic["code"].astype(str).str.zfill(6)
            return self._basic
        basic = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,market,list_date",
        )
        basic["code"] = basic["symbol"].map(_normalize_code)
        basic.to_csv(cache_path, index=False, encoding="utf-8-sig")
        self._basic = basic
        return basic

    def list_universe(self) -> pd.DataFrame:
        basic = self._load_basic()
        out = basic[["code", "name"]].copy()
        out["circ_mv"] = float("nan")
        out["price"] = float("nan")
        out["is_st"] = out["name"].str.contains("ST", case=False, na=False)
        max_stocks = int(self.data_cfg.get("max_stocks", 0) or 0)
        if max_stocks > 0:
            out = out.head(max_stocks)
        return out.reset_index(drop=True)

    def get_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = _normalize_code(code)
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        cache_path = self.cache_dir / "daily" / f"{code}.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = pd.read_csv(cache_path, dtype={"code": str})
            cached["date"] = pd.to_datetime(cached["date"])
            mask = (cached["date"] >= pd.Timestamp(start_date)) & (cached["date"] <= pd.Timestamp(end_date))
            return cached.loc[mask].reset_index(drop=True)

        df = self.pro.daily(ts_code=_ts_code(code), start_date=start, end_date=end)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"])
        # 换手率来自 daily_basic
        basic = self.pro.daily_basic(ts_code=_ts_code(code), start_date=start, end_date=end, fields="trade_date,turnover_rate")
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["trade_date"]),
                "code": code,
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["vol"], errors="coerce"),
                "amount": pd.to_numeric(df["amount"], errors="coerce") * 1000,
                "pct_chg": pd.to_numeric(df["pct_chg"], errors="coerce"),
            }
        )
        if basic is not None and not basic.empty:
            basic = basic.copy()
            basic["date"] = pd.to_datetime(basic["trade_date"])
            out = out.merge(basic[["date", "turnover_rate"]], on="date", how="left")
            out = out.rename(columns={"turnover_rate": "turnover"})
        else:
            out["turnover"] = float("nan")
        out = out.sort_values("date").reset_index(drop=True)
        out.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return out

    def get_fundamentals(self, codes: list[str] | None = None) -> pd.DataFrame:
        basic = self._load_basic()
        # 取最近一个交易日 daily_basic
        trade_cal = self.pro.trade_cal(exchange="SSE", is_open="1")
        last = trade_cal[trade_cal["cal_date"] <= pd.Timestamp.today().strftime("%Y%m%d")].iloc[-1]["cal_date"]
        db = self.pro.daily_basic(
            trade_date=last,
            fields="ts_code,trade_date,pe_ttm,pb,circ_mv,total_mv,turnover_rate",
        )
        db["code"] = db["ts_code"].str.slice(0, 6)
        merged = basic[["code", "name"]].merge(db, on="code", how="left")
        out = pd.DataFrame(
            {
                "code": merged["code"],
                "name": merged["name"],
                "pe_ttm": pd.to_numeric(merged.get("pe_ttm"), errors="coerce"),
                "pb": pd.to_numeric(merged.get("pb"), errors="coerce"),
                "circ_mv": pd.to_numeric(merged.get("circ_mv"), errors="coerce") / 10000.0,  # 万元 -> 亿元
                "total_mv": pd.to_numeric(merged.get("total_mv"), errors="coerce") / 10000.0,
                "revenue_yoy": float("nan"),
                "profit_yoy": float("nan"),
                "is_st": merged["name"].str.contains("ST", case=False, na=False),
            }
        )
        if codes is not None:
            code_set = {_normalize_code(c) for c in codes}
            out = out[out["code"].isin(code_set)]
        return out.reset_index(drop=True)
