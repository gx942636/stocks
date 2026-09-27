"""VPN 友好网络分流：保留系统/VPN 代理，仅让国内行情请求强制直连。"""
from __future__ import annotations

import os
import random
import time
from typing import Any
from urllib.parse import urlparse

# A 股行情常用域名（AKShare / 东方财富 / 新浪等）
DOMESTIC_DATA_HOSTS: tuple[str, ...] = (
    "eastmoney.com",
    ".eastmoney.com",
    "push2.eastmoney.com",
    "push2delay.eastmoney.com",
    "82.push2.eastmoney.com",
    "89.push2.eastmoney.com",
    "95.push2.eastmoney.com",
    "quote.eastmoney.com",
    "datacenter.eastmoney.com",
    "datacenter-web.eastmoney.com",
    "finance.eastmoney.com",
    "sina.com.cn",
    ".sina.com.cn",
    "finance.sina.com.cn",
    "vip.stock.finance.sina.com.cn",
    "hq.sinajs.cn",
    "cninfo.com.cn",
    ".cninfo.com.cn",
    "sse.com.cn",
    "szse.cn",
)

_APPLIED = False
_PATCHED = False
_ORIG_SESSION_REQUEST = None
_ORIG_REQUEST_WITH_RETRY = None


def _proxy_env_snapshot() -> dict[str, str]:
    keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    )
    return {k: os.environ[k] for k in keys if os.environ.get(k)}


def _merge_no_proxy(existing: str, hosts: tuple[str, ...]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for raw in (existing or "").split(","):
        item = raw.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(item)
    for host in hosts:
        key = host.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(host)
    return ",".join(parts)


def is_domestic_data_url(url: str) -> bool:
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    for item in DOMESTIC_DATA_HOSTS:
        key = item.lower().lstrip(".")
        if host == key or host.endswith("." + key):
            return True
    return False


def _direct_proxies() -> dict[str, None]:
    return {"http": None, "https": None}


def domestic_get(url: str, params: dict | None = None, timeout: float = 15):
    """国内行情强制直连 GET（不读系统/环境代理）。"""
    import requests

    with requests.Session() as session:
        session.trust_env = False
        return session.get(url, params=params, timeout=timeout, proxies=_direct_proxies())


def apply_vpn_friendly_split(enabled: bool = True) -> dict[str, Any]:
    """
    保留 HTTP(S)_PROXY，把国内行情域名追加进 NO_PROXY，
    并安装「仅国内 URL 强制直连」补丁。不做全局清空代理。
    """
    global _APPLIED
    result: dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "kept_proxy": False,
        "no_proxy": "",
        "probe_ok": None,
        "probe_detail": "",
        "patch_installed": False,
    }
    if not enabled:
        return result

    before = _proxy_env_snapshot()
    result["kept_proxy"] = any(
        k.lower() in ("http_proxy", "https_proxy", "all_proxy") for k in before
    )

    merged = _merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "", DOMESTIC_DATA_HOSTS)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    result["no_proxy"] = merged
    result["applied"] = True
    _APPLIED = True

    result["patch_installed"] = install_akshare_direct_patch()
    ok, detail = probe_eastmoney()
    result["probe_ok"] = ok
    result["probe_detail"] = detail
    return result


def ensure_vpn_split(enabled: bool = True) -> None:
    """幂等确保 NO_PROXY 与强制直连补丁生效。"""
    if not enabled:
        return
    if _APPLIED:
        merged = _merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "", DOMESTIC_DATA_HOSTS)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
        install_akshare_direct_patch()
        return
    apply_vpn_friendly_split(True)


def install_akshare_direct_patch() -> bool:
    """
    1) 补丁 requests.Session.request：国内行情 URL 强制直连
    2) 补丁 akshare.utils.request / func.request_with_retry：更多重试
       （func 模块在 import 时已绑定旧函数，必须两处都补）
    """
    global _PATCHED, _ORIG_SESSION_REQUEST, _ORIG_REQUEST_WITH_RETRY
    if _PATCHED:
        # 仍确保 func 模块引用也被更新（防止重复 import 顺序问题）
        try:
            import akshare.utils.func as ak_func
            import akshare.utils.request as ak_req

            if getattr(ak_func, "request_with_retry", None) is not ak_req.request_with_retry:
                ak_func.request_with_retry = ak_req.request_with_retry
        except Exception:
            pass
        return True

    import requests
    from requests.adapters import HTTPAdapter

    if _ORIG_SESSION_REQUEST is None:
        _ORIG_SESSION_REQUEST = requests.sessions.Session.request

        def _patched_session_request(self, method, url, *args, **kwargs):
            if is_domestic_data_url(str(url)):
                self.trust_env = False
                if "proxies" not in kwargs or kwargs.get("proxies") is None:
                    kwargs["proxies"] = _direct_proxies()
            return _ORIG_SESSION_REQUEST(self, method, url, *args, **kwargs)

        requests.sessions.Session.request = _patched_session_request  # type: ignore[method-assign]

    try:
        import akshare.utils.func as ak_func
        import akshare.utils.request as ak_req

        if _ORIG_REQUEST_WITH_RETRY is None:
            _ORIG_REQUEST_WITH_RETRY = ak_req.request_with_retry

            def _patched_request_with_retry(
                url,
                params=None,
                timeout: int = 15,
                max_retries: int = 5,
                base_delay: float = 1.2,
                random_delay_range=(0.8, 2.0),
            ):
                last_exception = None
                domestic = is_domestic_data_url(str(url))
                retries = max(max_retries, 5) if domestic else max_retries
                for attempt in range(retries):
                    try:
                        with requests.Session() as session:
                            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
                            session.mount("http://", adapter)
                            session.mount("https://", adapter)
                            if domestic:
                                session.trust_env = False
                                response = session.get(
                                    url,
                                    params=params,
                                    timeout=timeout,
                                    proxies=_direct_proxies(),
                                )
                            else:
                                response = session.get(url, params=params, timeout=timeout)
                            response.raise_for_status()
                            return response
                    except (requests.RequestException, ValueError) as e:
                        last_exception = e
                        if attempt < retries - 1:
                            delay = base_delay * (2**attempt) + random.uniform(*random_delay_range)
                            time.sleep(delay)
                raise last_exception

            ak_req.request_with_retry = _patched_request_with_retry

        # 关键：覆盖 func 里已绑定的旧引用
        ak_func.request_with_retry = ak_req.request_with_retry
    except Exception:
        pass

    _PATCHED = True
    return True


def probe_eastmoney(timeout: float = 8.0) -> tuple[bool, str]:
    """用与全市场接口同构的 clist 第一页 + 强制直连探测。"""
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "20",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f12,f14,f2",
    }
    try:
        resp = domestic_get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        data = resp.json()
        total = (((data or {}).get("data") or {}).get("total"))
        diff = (((data or {}).get("data") or {}).get("diff"))
        n = len(diff) if isinstance(diff, list) else 0
        if n <= 0 and not total:
            return False, "clist 返回空数据"
        return True, f"clist OK total={total} page_rows={n}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def format_vpn_fetch_error(exc: BaseException) -> str:
    """网络失败时的中文说明（不建议关 VPN，不建议清空全部代理）。"""
    return (
        "拉取 A 股行情失败（东财/新浪均未成功）。\n"
        "程序已保留 VPN/系统代理；并对国内行情域名使用强制直连（不经系统代理）。\n"
        "若仍失败，请在 VPN/Clash 客户端为以下域名设置 DIRECT，或开启「绕过中国大陆」：\n"
        "  *.eastmoney.com, push2.eastmoney.com, *.sina.com.cn\n"
        "（TUN 全局模式可能仍会拦截直连，必须在客户端加规则。）\n"
        f"原始错误: {type(exc).__name__}: {exc}"
    )
