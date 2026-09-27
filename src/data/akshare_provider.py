"""AKShare 数据适配（默认源）。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from .base import DataProvider
from .network_util import (
    domestic_get,
    ensure_vpn_split,
    format_vpn_fetch_error,
    install_akshare_direct_patch,
)


def _normalize_code(code: str) -> str:
    code = str(code).strip()
    if "." in code:
        code = code.split(".")[0]
    return code.zfill(6)


_EMPTY_DAILY_COLUMNS = [
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


def _sina_symbol(code: str) -> str:
    """新浪日线代码：沪市 sh，深市 sz。"""
    code = _normalize_code(code)
    if code.startswith(("5", "6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_DAILY_COLUMNS)


def _frame_from_sina(code: str, raw: pd.DataFrame) -> pd.DataFrame:
    """新浪日线映射到统一列。换手率由比例改为百分数，成交量改为「手」以便和东财缓存一致。"""
    close = pd.to_numeric(raw["close"], errors="coerce")
    volume = pd.to_numeric(raw["volume"], errors="coerce") / 100.0
    turnover = pd.to_numeric(raw["turnover"], errors="coerce") * 100.0 if "turnover" in raw.columns else float("nan")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"]),
            "code": code,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": close,
            "volume": volume,
            "amount": pd.to_numeric(raw["amount"], errors="coerce"),
            "turnover": turnover,
            "pct_chg": close.pct_change() * 100.0,
        }
    )


def _to_yi(value: float) -> float:
    """流通/总市值统一为亿元。原始常为元。"""
    if pd.isna(value):
        return float("nan")
    v = float(value)
    if v > 1e6:  # 按元
        return v / 1e8
    return v


class AkshareProvider(DataProvider):
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        super().__init__(config, project_root)
        self._spot: pd.DataFrame | None = None
        self._pause = float(self.data_cfg.get("request_pause", 0.15))
        self._adjust = self.data_cfg.get("adjust", "qfq")
        self._vpn_split = bool(self.data_cfg.get("vpn_split_domestic", True))
        self.daily_fetch_failures = 0
        self._logged_sina_daily = False
        self._logged_sina_fallback = False
        self._logged_em_daily_error = False

    def _prepare_network(self) -> None:
        ensure_vpn_split(self._vpn_split)
        if self._vpn_split:
            install_akshare_direct_patch()

    def _import_ak(self):
        self._prepare_network()
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("请先安装 akshare: pip install akshare") from exc
        return ak

    def _normalize_spot_em(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "code": raw["代码"].map(_normalize_code),
                "name": raw["名称"].astype(str),
                "price": pd.to_numeric(raw["最新价"], errors="coerce"),
                "pct_chg": pd.to_numeric(raw["涨跌幅"], errors="coerce"),
                "volume": pd.to_numeric(raw["成交量"], errors="coerce"),
                "amount": pd.to_numeric(raw["成交额"], errors="coerce"),
                "volume_ratio": pd.to_numeric(raw["量比"], errors="coerce") if "量比" in raw.columns else float("nan"),
                "turnover": pd.to_numeric(raw["换手率"], errors="coerce") if "换手率" in raw.columns else float("nan"),
                "pe_ttm": pd.to_numeric(raw.get("市盈率-动态", raw.get("市盈率")), errors="coerce"),
                "pb": pd.to_numeric(raw["市净率"], errors="coerce") if "市净率" in raw.columns else float("nan"),
                "total_mv": raw["总市值"].map(_to_yi) if "总市值" in raw.columns else float("nan"),
                "circ_mv": raw["流通市值"].map(_to_yi) if "流通市值" in raw.columns else float("nan"),
            }
        )
        df["is_st"] = df["name"].str.contains("ST", case=False, na=False)
        df["revenue_yoy"] = float("nan")
        df["profit_yoy"] = float("nan")
        return df

    def _load_spot_from_sina(self) -> pd.DataFrame:
        """东财失败时的新浪备用源（强制直连，保留市值/换手等字段）。"""
        import re

        from akshare.stock.cons import (
            zh_sina_a_stock_count_url,
            zh_sina_a_stock_payload,
            zh_sina_a_stock_url,
        )
        from akshare.utils import demjson
        from akshare.utils.tqdm import get_tqdm

        count_resp = domestic_get(zh_sina_a_stock_count_url, timeout=15)
        total = int(re.findall(r"\d+", count_resp.text)[0])
        page_count = max(1, (total + 79) // 80)

        rows: list[dict[str, Any]] = []
        payload = zh_sina_a_stock_payload.copy()
        tqdm = get_tqdm()
        for page in tqdm(range(1, page_count + 1), leave=False, desc="sina spot"):
            payload.update({"page": page})
            r = domestic_get(zh_sina_a_stock_url, params=payload, timeout=15)
            data_json = demjson.decode(r.text)
            for item in data_json:
                code = _normalize_code(item.get("code", ""))
                name = str(item.get("name", ""))
                # 新浪 nmc/mktcap 多为万元
                nmc = pd.to_numeric(item.get("nmc"), errors="coerce")
                mkt = pd.to_numeric(item.get("mktcap"), errors="coerce")
                circ_yi = float(nmc) / 10000.0 if pd.notna(nmc) else float("nan")
                total_yi = float(mkt) / 10000.0 if pd.notna(mkt) else float("nan")
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "price": pd.to_numeric(item.get("trade"), errors="coerce"),
                        "pct_chg": pd.to_numeric(item.get("changepercent"), errors="coerce"),
                        "volume": pd.to_numeric(item.get("volume"), errors="coerce"),
                        "amount": pd.to_numeric(item.get("amount"), errors="coerce"),
                        "volume_ratio": float("nan"),
                        "turnover": pd.to_numeric(item.get("turnoverratio"), errors="coerce"),
                        "pe_ttm": pd.to_numeric(item.get("per"), errors="coerce"),
                        "pb": pd.to_numeric(item.get("pb"), errors="coerce"),
                        "total_mv": total_yi,
                        "circ_mv": circ_yi,
                        "is_st": "ST" in name.upper() or "退" in name,
                        "revenue_yoy": float("nan"),
                        "profit_yoy": float("nan"),
                    }
                )
        return pd.DataFrame(rows)

    def _load_spot(self) -> pd.DataFrame:
        if self._spot is not None:
            return self._spot
        cache_path = self.cache_dir / "spot_em.csv"
        # 当天缓存可复用；简单按文件是否存在且较新（6小时）
        if cache_path.exists():
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime < 6 * 3600:
                self._spot = pd.read_csv(cache_path, dtype={"code": str})
                self._spot["code"] = self._spot["code"].map(_normalize_code)
                return self._spot

        self._prepare_network()
        last_exc: Exception | None = None
        try:
            df = self._load_spot_from_sina()
            print("行情来源: 新浪财经")
        except Exception as exc:
            last_exc = exc
            print(f"新浪列表失败，改走东方财富: {type(exc).__name__}")
            try:
                ak = self._import_ak()
                raw = ak.stock_zh_a_spot_em()
                df = self._normalize_spot_em(raw)
                print("行情来源: 东方财富备用 stock_zh_a_spot_em")
            except Exception as em_exc:
                raise RuntimeError(format_vpn_fetch_error(em_exc)) from em_exc

        if df is None or df.empty:
            raise RuntimeError(format_vpn_fetch_error(last_exc or RuntimeError("空行情")))

        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        self._spot = df
        return df

    def list_universe(self) -> pd.DataFrame:
        spot = self._load_spot()
        max_stocks = int(self.data_cfg.get("max_stocks", 0) or 0)
        out = spot[["code", "name", "circ_mv", "price", "is_st"]].copy()
        if max_stocks > 0:
            # 优先中等流通市值，更贴近策略区间
            out = out.sort_values("circ_mv", ascending=True)
            mid = out[(out["circ_mv"] >= 40) & (out["circ_mv"] <= 350)]
            if len(mid) >= max_stocks:
                out = mid.head(max_stocks)
            else:
                out = out.head(max_stocks)
        return out.reset_index(drop=True)

    def get_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = _normalize_code(code)
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        cache_path = self.cache_dir / "daily" / f"{code}.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached = None
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, dtype={"code": str})
                cached["date"] = pd.to_datetime(cached["date"])
            except Exception:
                cached = None

        need_fetch = True
        if cached is not None and not cached.empty:
            cmin, cmax = cached["date"].min(), cached["date"].max()
            if cmin <= pd.Timestamp(start_date) and cmax >= pd.Timestamp(end_date):
                need_fetch = False

        if need_fetch:
            time.sleep(self._pause)
            df = self._pull_daily(code, start, end)
            if df is None or df.empty:
                if cached is None or cached.empty:
                    self.daily_fetch_failures += 1
                    return _empty_daily()
            else:
                if cached is not None and not cached.empty:
                    df = (
                        pd.concat([cached, df], ignore_index=True)
                        .drop_duplicates(subset=["date"], keep="last")
                        .sort_values("date")
                    )
                df.to_csv(cache_path, index=False, encoding="utf-8-sig")
                cached = df

        assert cached is not None
        mask = (cached["date"] >= pd.Timestamp(start_date)) & (cached["date"] <= pd.Timestamp(end_date))
        out = cached.loc[mask].copy()
        out["code"] = code
        return out.reset_index(drop=True)

    def _pull_daily(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        """先新浪，这一只失败或为空时再试东财。"""
        frame = self._fetch_sina_daily(code, start, end)
        if frame is not None and not frame.empty:
            return frame
        if not self._logged_sina_fallback:
            self._logged_sina_fallback = True
            print("新浪日线失败，该股改走东方财富")
        try:
            frame = self._fetch_em_daily(code, start, end)
        except Exception as exc:
            self._note_em_daily_error(exc)
            return None
        return frame

    def _note_em_daily_error(self, exc: Exception) -> None:
        if self._logged_em_daily_error:
            return
        self._logged_em_daily_error = True
        msg = str(exc).lower()
        if "proxy" in msg or "proxyerror" in type(exc).__name__.lower():
            print(format_vpn_fetch_error(exc))
        else:
            print(f"东方财富日线也失败: {type(exc).__name__}")

    def _fetch_em_daily(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        ak = self._import_ak()
        raw = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=self._adjust,
        )
        if raw is None or raw.empty:
            return None
        return pd.DataFrame(
            {
                "date": pd.to_datetime(raw["日期"]),
                "code": code,
                "open": pd.to_numeric(raw["开盘"], errors="coerce"),
                "high": pd.to_numeric(raw["最高"], errors="coerce"),
                "low": pd.to_numeric(raw["最低"], errors="coerce"),
                "close": pd.to_numeric(raw["收盘"], errors="coerce"),
                "volume": pd.to_numeric(raw["成交量"], errors="coerce"),
                "amount": pd.to_numeric(raw["成交额"], errors="coerce"),
                "turnover": pd.to_numeric(raw["换手率"], errors="coerce"),
                "pct_chg": pd.to_numeric(raw["涨跌幅"], errors="coerce"),
            }
        )

    def _fetch_sina_daily(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        ak = self._import_ak()
        adjust = self._adjust if self._adjust in ("qfq", "hfq", "") else "qfq"
        try:
            raw = ak.stock_zh_a_daily(
                symbol=_sina_symbol(code),
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            if raw is None or raw.empty or "close" not in getattr(raw, "columns", []):
                return None
            frame = _frame_from_sina(code, raw)
        except Exception:
            return None
        if not self._logged_sina_daily:
            self._logged_sina_daily = True
            print("日线来源: 新浪财经 stock_zh_a_daily")
        return frame

    def get_fundamentals(self, codes: list[str] | None = None) -> pd.DataFrame:
        spot = self._load_spot()
        cols = [
            "code",
            "name",
            "pe_ttm",
            "pb",
            "circ_mv",
            "total_mv",
            "revenue_yoy",
            "profit_yoy",
            "is_st",
            "volume_ratio",
            "turnover",
            "price",
            "pct_chg",
        ]
        df = spot[cols].copy()
        if codes is not None:
            code_set = {_normalize_code(c) for c in codes}
            df = df[df["code"].isin(code_set)]
        # 尝试补充盈利增速（失败则保持 NaN）
        df = self._try_fill_growth(df)
        return df.reset_index(drop=True)

    def _try_fill_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """只用本地增速缓存。不主动请求东方财富。"""
        cache_path = self.cache_dir / "growth_yoy.csv"
        if cache_path.exists():
            try:
                growth = pd.read_csv(cache_path, dtype={"code": str})
                growth["code"] = growth["code"].map(_normalize_code)
                return df.drop(columns=["revenue_yoy", "profit_yoy"], errors="ignore").merge(
                    growth, on="code", how="left"
                )
            except Exception:
                pass
        return df
