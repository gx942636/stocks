"""统一数据源接口与工厂。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


OHLCV_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_chg",
]

FUND_COLUMNS = [
    "code",
    "name",
    "pe_ttm",
    "pb",
    "circ_mv",
    "total_mv",
    "revenue_yoy",
    "profit_yoy",
    "is_st",
]


class DataProvider(ABC):
    """行情 / 基本面统一接口。"""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.data_cfg = config.get("data", {})
        self.project_root = project_root
        self.cache_dir = project_root / self.data_cfg.get("cache_dir", "data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def list_universe(self) -> pd.DataFrame:
        """返回候选股票池，至少含 code, name。"""

    @abstractmethod
    def get_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """日线，统一列名见 OHLCV_COLUMNS。日期为 datetime64。"""

    @abstractmethod
    def get_fundamentals(self, codes: list[str] | None = None) -> pd.DataFrame:
        """基本面截面，统一列名见 FUND_COLUMNS。circ_mv 单位：亿元。"""


def get_provider(config: dict[str, Any], project_root: Path) -> DataProvider:
    source = str(config.get("data", {}).get("source", "akshare")).lower()
    if source == "akshare":
        from .akshare_provider import AkshareProvider

        return AkshareProvider(config, project_root)
    if source == "tushare":
        from .tushare_provider import TushareProvider

        return TushareProvider(config, project_root)
    if source == "csv":
        from .csv_provider import CsvProvider

        return CsvProvider(config, project_root)
    raise ValueError(f"未知数据源: {source}，可选 akshare / tushare / csv")
