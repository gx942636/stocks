"""本地 CSV 数据适配（离线回测）。

目录结构（config.data.csv_dir）：
  stock_list.csv   列: code,name[,circ_mv,is_st]
  fundamentals.csv 列: code,name,pe_ttm,pb,circ_mv,total_mv,revenue_yoy,profit_yoy,is_st
  daily/{code}.csv 列: date,open,high,low,close,volume,amount,turnover,pct_chg
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import DataProvider


def _normalize_code(code: str) -> str:
    code = str(code).strip()
    if "." in code:
        code = code.split(".")[0]
    return code.zfill(6)


class CsvProvider(DataProvider):
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        super().__init__(config, project_root)
        self.csv_dir = project_root / self.data_cfg.get("csv_dir", "data_cache/csv")
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        (self.csv_dir / "daily").mkdir(parents=True, exist_ok=True)

    def list_universe(self) -> pd.DataFrame:
        path = self.csv_dir / "stock_list.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}，请按 README 准备 CSV 数据")
        df = pd.read_csv(path, dtype={"code": str})
        df["code"] = df["code"].map(_normalize_code)
        if "name" not in df.columns:
            df["name"] = df["code"]
        if "is_st" not in df.columns:
            df["is_st"] = df["name"].astype(str).str.contains("ST", case=False, na=False)
        if "circ_mv" not in df.columns:
            df["circ_mv"] = float("nan")
        if "price" not in df.columns:
            df["price"] = float("nan")
        max_stocks = int(self.data_cfg.get("max_stocks", 0) or 0)
        if max_stocks > 0:
            df = df.head(max_stocks)
        return df[["code", "name", "circ_mv", "price", "is_st"]].reset_index(drop=True)

    def get_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = _normalize_code(code)
        path = self.csv_dir / "daily" / f"{code}.csv"
        if not path.exists():
            return pd.DataFrame(columns=["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"])
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = code
        for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]:
            if col not in df.columns:
                df[col] = float("nan")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if df["pct_chg"].isna().all():
            df["pct_chg"] = df["close"].pct_change() * 100
        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        return df.loc[mask, ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]].reset_index(drop=True)

    def get_fundamentals(self, codes: list[str] | None = None) -> pd.DataFrame:
        path = self.csv_dir / "fundamentals.csv"
        if not path.exists():
            # 退化：仅用股票列表
            uni = self.list_universe()
            out = pd.DataFrame(
                {
                    "code": uni["code"],
                    "name": uni["name"],
                    "pe_ttm": float("nan"),
                    "pb": float("nan"),
                    "circ_mv": uni.get("circ_mv", float("nan")),
                    "total_mv": float("nan"),
                    "revenue_yoy": float("nan"),
                    "profit_yoy": float("nan"),
                    "is_st": uni["is_st"],
                }
            )
        else:
            out = pd.read_csv(path, dtype={"code": str})
            out["code"] = out["code"].map(_normalize_code)
            for col in ["pe_ttm", "pb", "circ_mv", "total_mv", "revenue_yoy", "profit_yoy"]:
                if col not in out.columns:
                    out[col] = float("nan")
            if "is_st" not in out.columns:
                out["is_st"] = out.get("name", pd.Series(dtype=str)).astype(str).str.contains("ST", case=False, na=False)
            if "name" not in out.columns:
                out["name"] = out["code"]
        if codes is not None:
            code_set = {_normalize_code(c) for c in codes}
            out = out[out["code"].isin(code_set)]
        return out.reset_index(drop=True)
