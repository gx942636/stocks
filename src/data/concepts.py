"""入选股的行业与题材。只在筛选之后按代码补拉，并缓存成功结果。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_NOT_THEME = {
    "沪股通",
    "深股通",
    "融资融券",
    "转融券标的",
    "沪深300",
    "上证50",
    "上证180",
    "上证380",
    "中证100",
    "中证500",
    "中证800",
    "中证1000",
    "中证2000",
    "深证100",
    "深证成指",
    "创业板指",
    "科创50",
    "北证50",
    "中盘股",
    "大盘股",
    "小盘股",
    "微盘股",
    "MSCI中国",
    "富时罗素",
    "标普道琼斯中国",
    "证金持股",
}

_EMPTY = {"ok": False, "industry": "", "themes": []}


def parse_theme_rows(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """从东财板块列表拆出行业和题材。指数、通、融资融券和地域板块不算题材。"""
    industries: list[str] = []
    themes: list[str] = []
    seen: set[str] = set()
    for row in rows or []:
        name = str(row.get("BOARD_NAME") or "").strip()
        if not name:
            continue
        kind = str(row.get("BOARD_TYPE") or "").strip()
        if kind == "行业":
            if not name.startswith("其他"):
                industries.append(name)
            continue
        if kind == "板块" or name.endswith("板块") or name in _NOT_THEME:
            continue
        if name in seen:
            continue
        seen.add(name)
        themes.append(name)
    industry = max(industries, key=len) if industries else ""
    return {"ok": True, "industry": industry, "themes": themes}


def _cache_path(cache_dir: Path, code: str) -> Path:
    return cache_dir / "concepts" / f"{str(code).zfill(6)}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    themes = data.get("themes") or []
    if not isinstance(themes, list):
        themes = []
    return {
        "ok": True,
        "industry": str(data.get("industry") or ""),
        "themes": [str(item) for item in themes if str(item).strip()],
    }


def _fetch_rows(code: str) -> list[dict[str, Any]] | None:
    import requests

    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    params = {
        "reportName": "RPT_F10_CORETHEME_BOARDTYPE",
        "columns": "SECURITY_CODE,BOARD_NAME,BOARD_TYPE,BOARD_CODE",
        "filter": f'(SECURITY_CODE="{str(code).zfill(6)}")',
        "pageNumber": "1",
        "pageSize": "200",
        "source": "HSF10",
        "client": "PC",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/",
    }
    try:
        with requests.Session() as session:
            session.trust_env = False
            resp = session.get(
                url,
                params=params,
                timeout=15,
                proxies={"http": None, "https": None},
                headers=headers,
            )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("success"):
        return None
    rows = ((payload.get("result") or {}).get("data")) or []
    if not isinstance(rows, list):
        return None
    return rows


def fetch_stock_profile(code: str, cache_dir: Path, pause: float = 0.0) -> dict[str, Any]:
    """返回行业和题材。失败时 ok 为 False，调用方显示「题材暂缺」且不剔除该股。"""
    path = _cache_path(cache_dir, code)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    if pause > 0:
        time.sleep(pause)
    rows = _fetch_rows(code)
    if rows is None:
        return dict(_EMPTY)
    profile = parse_theme_rows(rows)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return profile
