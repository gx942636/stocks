"""中线低位企稳 / 短线两种方案的本地网页入口。

启动：streamlit run app.py
选定一个假设买入日，结果只在页面上查看。
"""
from __future__ import annotations

import copy
import html
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from main import load_config
from src.data.base import get_provider
from src.data.concepts import fetch_stock_profile
from src.data.network_util import apply_vpn_friendly_split, install_akshare_direct_patch, use_sina_quote_source
from src.pipeline import (
    build_fund_map,
    collect_stock_data,
    resolve_asof,
    screen_asof,
    screen_short_asof,
)
from src.strategy.style import active_strategy, fetch_strategy, resolve_style

CANDIDATE_LABELS = [
    ("date", "买入日"),
    ("code", "代码"),
    ("name", "名称"),
    ("signal_close", "收盘价"),
    ("drawdown_60", "60日回撤(%)"),
    ("low_20", "20日低点"),
    ("low_dist_pct", "距低点(%)"),
    ("ret_5d", "近5日涨幅(%)"),
    ("amount_yi", "成交额中位数(亿)"),
    ("pe_ttm", "PE"),
    ("pb", "PB"),
    ("industry", "行业"),
]

TREND_LABELS = [
    ("date", "买入日"),
    ("code", "代码"),
    ("name", "名称"),
    ("signal_close", "收盘价"),
    ("ret_20d", "20日涨幅(%)"),
    ("drawdown_10", "距10日高点(%)"),
    ("pct_chg", "当日涨幅(%)"),
    ("amount_ratio", "量比"),
    ("close_pos_pct", "收盘位置(%)"),
    ("resist_price", "压力价"),
    ("resist_room", "距压力(%)"),
    ("industry", "行业"),
]

PULL_LABELS = [
    ("date", "买入日"),
    ("code", "代码"),
    ("name", "名称"),
    ("signal_close", "收盘价"),
    ("drawdown_10", "10日回撤(%)"),
    ("low_8", "8日低点"),
    ("low_dist_8", "距8日低点(%)"),
    ("ret_3d", "近3日涨幅(%)"),
    ("amount_ratio", "量比"),
    ("close_pos_pct", "收盘位置(%)"),
    ("resist_price", "压力价"),
    ("resist_room", "距压力(%)"),
    ("industry", "行业"),
]


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def ensure_network(config: dict[str, Any]) -> None:
    """每个会话只做一次国内行情直连设置。"""
    if "network_note" in st.session_state:
        return
    data_cfg = config.get("data", {}) or {}
    source = str(data_cfg.get("source", "akshare")).lower()
    note = ""
    warn = False
    if source == "akshare" and bool(data_cfg.get("vpn_split_domestic", True)):
        try:
            split_info = apply_vpn_friendly_split(True)
            install_akshare_direct_patch()
            note = "网络分流：已保留 VPN/系统代理，国内行情强制直连。行情主源是新浪，失败时再试东方财富。"
            if split_info.get("probe_ok") is False:
                note += " 东财探测失败，备用源可能不可用。"
                warn = True
        except Exception as exc:
            note = f"网络分流设置失败：{exc}"
            warn = True
    st.session_state["network_note"] = note
    st.session_state["network_warn"] = warn


def validate_form(form: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if form["max_stocks"] <= 0:
        errors.append("股票数上限必须为正数。")
    return errors


def apply_form(base: dict[str, Any], form: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["style"] = form["style"]
    cfg["as_of_date"] = form["as_of_date"].strftime("%Y-%m-%d")
    cfg.setdefault("data", {})
    cfg["data"]["max_stocks"] = int(form["max_stocks"])
    return cfg


def _fill_circ_mv(circ_mv_map: dict[str, float], fund_map: dict[str, dict[str, Any]]) -> None:
    for code, fund in fund_map.items():
        current = circ_mv_map.get(code)
        missing = code not in circ_mv_map or pd.isna(current)
        if missing and pd.notna(fund.get("circ_mv")):
            circ_mv_map[code] = float(fund["circ_mv"])


def run_screen_job(config: dict[str, Any], on_progress) -> dict[str, Any]:
    as_of = resolve_asof(config)
    style = resolve_style(config)
    strategy = active_strategy(config)
    fetching = fetch_strategy(config)
    data_cfg = config.get("data", {}) or {}
    cache_key = (
        as_of,
        str(data_cfg.get("source", "akshare")).lower(),
        int(data_cfg.get("max_stocks", 0) or 0),
        "sina" if use_sina_quote_source() else "eastmoney",
        "raw-v1",
        int(fetching.get("history_calendar_days", 140)),
        int(fetching.get("pullback_days", 60)),
        float(fetching.get("circ_mv_min_yi", 30)),
        bool(fetching.get("exclude_st", True)),
        bool(fetching.get("exclude_kcb", True)),
        bool(fetching.get("exclude_cyb", True)),
    )

    provider = get_provider(config, PROJECT_ROOT)
    cache = st.session_state.get("market_cache")
    from_cache = isinstance(cache, dict) and cache.get("key") == cache_key
    if from_cache:
        daily_map = cache["daily_map"]
        name_map = cache["name_map"]
        circ_mv_map = cache["circ_mv_map"]
        fund_map = cache["fund_map"]
    else:
        universe = provider.list_universe()
        daily_map, name_map, circ_mv_map = collect_stock_data(
            provider, universe, as_of, as_of, fetching, progress=on_progress
        )
        on_progress(0, 0, "读取基本面")
        fund_map = build_fund_map(provider, list(daily_map.keys()))
        _fill_circ_mv(circ_mv_map, fund_map)
        fetch_failures = int(getattr(provider, "daily_fetch_failures", 0) or 0)
        if not (len(daily_map) == 0 and fetch_failures > 0):
            st.session_state["market_cache"] = {
                "key": cache_key,
                "daily_map": daily_map,
                "name_map": name_map,
                "circ_mv_map": circ_mv_map,
                "fund_map": fund_map,
            }

    if style == "short":
        candidates = screen_short_asof(
            daily_map,
            circ_mv_map,
            name_map,
            as_of,
            strategy,
            progress=on_progress,
        )
    else:
        candidates = screen_asof(
            daily_map,
            circ_mv_map,
            name_map,
            as_of,
            strategy,
            fund_map,
            progress=on_progress,
        )
    fetch_failures = 0 if from_cache else int(getattr(provider, "daily_fetch_failures", 0) or 0)
    return {
        "as_of": as_of,
        "style": style,
        "from_cache": from_cache,
        "n_daily": len(daily_map),
        "n_candidates": 0 if candidates is None or candidates.empty else len(candidates),
        "daily_fetch_failures": fetch_failures,
        "candidates": candidates,
        "klines": _candidate_klines(daily_map, candidates),
        "cache_dir": str(PROJECT_ROOT / str(data_cfg.get("cache_dir", "data_cache"))),
    }


def _candidate_klines(
    daily_map: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    bars: int = 120,
) -> dict[str, pd.DataFrame]:
    """每只入选股保留最近一段日线，供 K 线使用，不再额外请求行情。"""
    out: dict[str, pd.DataFrame] = {}
    if candidates is None or candidates.empty:
        return out
    keep = ["date", "open", "high", "low", "close", "volume"]
    for code in candidates["code"].astype(str):
        code = code.zfill(6)
        daily = daily_map.get(code)
        if daily is None or daily.empty:
            continue
        tail = daily.sort_values("date").tail(bars)
        cols = [col for col in keep if col in tail.columns]
        out[code] = tail[cols].copy()
    return out


def _labeled_frame(df: pd.DataFrame, labels: list[tuple[str, str]]) -> pd.DataFrame:
    view = df.copy()
    cols = [key for key, _ in labels if key in view.columns]
    rename = {key: label for key, label in labels if key in view.columns}
    view = view[cols].rename(columns=rename)
    for col in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[col]):
            view[col] = view[col].dt.strftime("%Y-%m-%d")
    return view


TERMINAL_CSS = """
<style>
footer, [data-testid="stFooter"] {visibility: hidden; height: 0;}
[data-testid="stDecoration"] {display: none;}
section[data-testid="stSidebar"] {border-right: 1px solid #2b3139;}
.term-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  padding: 2px 0 14px;
  margin-bottom: 8px;
  border-bottom: 1px solid #2b3139;
}
.term-kicker {
  color: #7dd3fc;
  letter-spacing: 0.18em;
  font-size: 11px;
}
.term-title {
  font-size: 28px;
  font-weight: 650;
  color: #f5f7fa;
  line-height: 1.2;
}
.term-title span {color: #f0b90b; font-weight: 500;}
.term-pills {display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end;}
.pill {
  border: 1px solid #2b3139;
  background: #161a1e;
  color: #eaecef;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
}
.side-label, .section-label {
  color: #f0b90b;
  font-size: 12px;
  letter-spacing: 0.14em;
  margin: 14px 0 6px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 8px 0 12px;
}
.metric-card {
  background: #161a1e;
  border: 1px solid #2b3139;
  border-radius: 10px;
  padding: 14px 16px 12px;
  min-height: 92px;
}
.metric-label {
  color: #8b949e;
  font-size: 12px;
  letter-spacing: 0.08em;
}
.metric-value {
  font-size: 26px;
  font-variant-numeric: tabular-nums;
  margin-top: 6px;
  color: #eaecef;
}
.metric-card.up {border-top: 2px solid #f6465d;}
.metric-card.up .metric-value {color: #f6465d;}
.metric-card.down {border-top: 2px solid #0ecb81;}
.metric-card.down .metric-value {color: #0ecb81;}
.empty-board {
  border: 1px dashed #2b3139;
  background: linear-gradient(180deg, rgba(240,185,11,0.06), transparent 46%);
  border-radius: 12px;
  padding: 48px 28px;
  margin-top: 16px;
}
.empty-title {
  color: #f0b90b;
  letter-spacing: 0.16em;
  font-size: 13px;
}
.empty-board p {color: #c9d1d9; margin: 10px 0 0;}
[data-testid="stDataFrame"] {border: 1px solid #2b3139; border-radius: 8px;}
.tag-row {display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 12px;}
.tag {
  border: 1px solid #2b3139;
  background: #1c2127;
  color: #eaecef;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 12px;
}
.tag.industry {border-color: #f0b90b; color: #f0b90b;}
.tag.missing {color: #8b949e;}
@media (max-width: 900px) {
  .metric-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .term-head {flex-direction: column; align-items: flex-start;}
}
</style>
"""


def _inject_css() -> None:
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


def _section(title: str) -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


def _side_label(title: str) -> None:
    st.markdown(f'<div class="side-label">{title}</div>', unsafe_allow_html=True)


def _tone(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _metric_card(label: str, value: str, tone: str) -> str:
    return (
        f'<div class="metric-card {tone}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f"</div>"
    )


def _render_header(as_of: date, style: str) -> None:
    if style == "short":
        kicker = "SHORT-TERM"
        title = "短线 <span>两种方案</span>"
    else:
        kicker = "A-SHARE RESEARCH"
        title = "中线低位 <span>企稳</span>"
    st.markdown(
        f"""
        <div class="term-head">
          <div>
            <div class="term-kicker">{kicker}</div>
            <div class="term-title">{title}</div>
          </div>
          <div class="term-pills">
            <span class="pill">假设买入日 {as_of:%Y-%m-%d}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty() -> None:
    st.markdown(
        """
        <div class="empty-board">
          <div class="empty-title">等待指令</div>
          <p>在左侧选定策略和假设买入日，然后开始筛选。阈值在 config.yaml，结果只在本页查看。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_equity(equity: pd.DataFrame) -> None:
    chart_df = equity.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.rename(columns={"equity": "权益"})
    chart = (
        alt.Chart(chart_df)
        .mark_area(
            line={"color": "#f0b90b", "strokeWidth": 2},
            color="rgba(240,185,11,0.18)",
        )
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d", grid=False)),
            y=alt.Y("权益:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(format=",.0f")),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("权益:Q", title="权益", format=",.2f"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
        .configure_axis(labelColor="#8b949e", gridColor="#2b3139", domainColor="#2b3139")
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def _fmt_num(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _num_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return number


def _resist_room_text(row: pd.Series) -> str:
    state = _num_or_zero(row.get("resist_state"))
    if state == 1:
        return "已突破"
    return _fmt_num(row.get("resist_room"), 1, "%")


def _candidate_view(candidates: pd.DataFrame, profiles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    view = candidates.copy()
    view["code"] = view["code"].astype(str).str.zfill(6)
    view["amount_yi"] = pd.to_numeric(view.get("amount_median"), errors="coerce") / 1e8
    view["industry"] = view["code"].map(
        lambda code: (profiles.get(code) or {}).get("industry") or "—"
    )
    if "close_pos" in view.columns:
        view["close_pos_pct"] = pd.to_numeric(view["close_pos"], errors="coerce") * 100
    for col, digits in (
        ("signal_close", 2),
        ("drawdown_60", 1),
        ("low_20", 2),
        ("low_dist_pct", 1),
        ("ret_5d", 1),
        ("ret_20d", 1),
        ("ret_3d", 1),
        ("drawdown_10", 1),
        ("low_8", 2),
        ("low_dist_8", 1),
        ("pct_chg", 2),
        ("amount_ratio", 2),
        ("close_pos_pct", 0),
        ("resist_price", 2),
        ("resist_room", 1),
        ("amount_yi", 2),
        ("pe_ttm", 1),
        ("pb", 2),
    ):
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").round(digits)
    return view


def _load_profiles(codes: list[str], cache_dir: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    total = len(codes)
    if total == 0:
        return profiles
    status = st.empty()
    for index, code in enumerate(codes, start=1):
        status.caption(f"读取题材：{index} / {total}")
        cached = (cache_dir / "concepts" / f"{code}.json").exists()
        profiles[code] = fetch_stock_profile(code, cache_dir, pause=0.0 if cached else 0.08)
    status.empty()
    return profiles


def _theme_html(profile: dict[str, Any]) -> str:
    if not profile or not profile.get("ok"):
        return '<div class="tag-row"><span class="tag missing">题材暂缺</span></div>'
    tags = []
    industry = str(profile.get("industry") or "").strip()
    if industry:
        tags.append(f'<span class="tag industry">{html.escape(industry)}</span>')
    themes = profile.get("themes") or []
    if not themes and not industry:
        tags.append('<span class="tag missing">暂无公开题材</span>')
    for name in themes:
        tags.append(f'<span class="tag">{html.escape(str(name))}</span>')
    return f'<div class="tag-row">{"".join(tags)}</div>'


def _render_kline(daily: pd.DataFrame, mark_date: Any) -> None:
    chart_df = daily.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in chart_df.columns:
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
    chart_df = chart_df.dropna(subset=["date", "open", "high", "low", "close"])
    if chart_df.empty:
        st.caption("没有足够的日线画出 K 线。")
        return
    chart_df["实体收盘"] = chart_df["close"]
    flat = chart_df["实体收盘"] == chart_df["open"]
    chart_df.loc[flat, "实体收盘"] = chart_df.loc[flat, "open"] * 1.0008
    chart_df["方向"] = (chart_df["close"] >= chart_df["open"]).map({True: "上涨", False: "下跌"})
    color_scale = alt.Scale(domain=["上涨", "下跌"], range=["#f6465d", "#0ecb81"])
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%m-%d", labelAngle=-40, grid=False))
    price_tip = [
        alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
        alt.Tooltip("open:Q", title="开盘", format=".2f"),
        alt.Tooltip("high:Q", title="最高", format=".2f"),
        alt.Tooltip("low:Q", title="最低", format=".2f"),
        alt.Tooltip("close:Q", title="收盘", format=".2f"),
    ]
    wick = (
        alt.Chart(chart_df)
        .mark_rule()
        .encode(
            x=x,
            y=alt.Y("low:Q", title=None, scale=alt.Scale(zero=False)),
            y2="high:Q",
            color=alt.Color("方向:N", scale=color_scale, legend=None),
            tooltip=price_tip,
        )
    )
    body = (
        alt.Chart(chart_df)
        .mark_bar(size=5)
        .encode(
            x=x,
            y="open:Q",
            y2="实体收盘:Q",
            color=alt.Color("方向:N", scale=color_scale, legend=None),
            tooltip=price_tip,
        )
    )
    price = wick + body
    mark = pd.to_datetime(mark_date, errors="coerce")
    if pd.notna(mark):
        rule = (
            alt.Chart(pd.DataFrame({"date": [mark]}))
            .mark_rule(color="#f0b90b", strokeDash=[4, 3], strokeWidth=1.4)
            .encode(x="date:T")
        )
        price = price + rule
    volume = (
        alt.Chart(chart_df)
        .mark_bar(size=5)
        .encode(
            x=x,
            y=alt.Y("volume:Q", title=None),
            color=alt.Color("方向:N", scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("volume:Q", title="成交量", format=",.0f"),
            ],
        )
    )
    chart = (
        alt.vconcat(price.properties(height=280), volume.properties(height=80))
        .resolve_scale(x="shared", color="independent")
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
        .configure_axis(labelColor="#8b949e", gridColor="#2b3139", domainColor="#2b3139")
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def _render_stock_cards(
    candidates: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    klines: dict[str, pd.DataFrame],
    section: str | None = "个股详情",
) -> None:
    if section:
        _section(section)
        st.caption("默认收起。黄色竖线是假设买入日。")
    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        name = str(row.get("name") or code)
        profile = profiles.get(code) or {}
        title = f"{code}  {name}    {_fmt_num(row.get('signal_close'), 2)} 元"
        with st.expander(title):
            amount = row.get("amount_median")
            try:
                amount_yi = float(amount) / 1e8
            except (TypeError, ValueError):
                amount_yi = float("nan")
            scheme = str(row.get("scheme") or "")
            if scheme == "趋势涨":
                cards = [
                    _metric_card("收盘价", _fmt_num(row.get("signal_close"), 2), "flat"),
                    _metric_card("20日涨幅", _fmt_num(row.get("ret_20d"), 1, "%"), _tone(_num_or_zero(row.get("ret_20d")))),
                    _metric_card("距10日高点", _fmt_num(row.get("drawdown_10"), 1, "%"), "flat"),
                    _metric_card("当日涨幅", _fmt_num(row.get("pct_chg"), 2, "%"), _tone(_num_or_zero(row.get("pct_chg")))),
                    _metric_card("量比", _fmt_num(row.get("amount_ratio"), 2), "flat"),
                    _metric_card("收盘位置", _fmt_num(row.get("close_pos_pct"), 0, "%"), "flat"),
                    _metric_card("压力价", _fmt_num(row.get("resist_price"), 2), "flat"),
                    _metric_card("距压力", _resist_room_text(row), "flat"),
                    _metric_card("20日成交额", _fmt_num(amount_yi, 2, "亿"), "flat"),
                ]
            elif scheme == "回踩止跌":
                cards = [
                    _metric_card("收盘价", _fmt_num(row.get("signal_close"), 2), "flat"),
                    _metric_card("10日回撤", _fmt_num(row.get("drawdown_10"), 1, "%"), "down"),
                    _metric_card("8日低点", _fmt_num(row.get("low_8"), 2), "flat"),
                    _metric_card("距低点", _fmt_num(row.get("low_dist_8"), 1, "%"), "flat"),
                    _metric_card("近3日涨幅", _fmt_num(row.get("ret_3d"), 1, "%"), _tone(_num_or_zero(row.get("ret_3d")))),
                    _metric_card("量比", _fmt_num(row.get("amount_ratio"), 2), "flat"),
                    _metric_card("收盘位置", _fmt_num(row.get("close_pos_pct"), 0, "%"), "flat"),
                    _metric_card("压力价", _fmt_num(row.get("resist_price"), 2), "flat"),
                    _metric_card("距压力", _resist_room_text(row), "flat"),
                    _metric_card("20日成交额", _fmt_num(amount_yi, 2, "亿"), "flat"),
                    _metric_card("当日涨幅", _fmt_num(row.get("pct_chg"), 2, "%"), _tone(_num_or_zero(row.get("pct_chg")))),
                ]
            else:
                cards = [
                    _metric_card("收盘价", _fmt_num(row.get("signal_close"), 2), "flat"),
                    _metric_card("60日回撤", _fmt_num(row.get("drawdown_60"), 1, "%"), "down"),
                    _metric_card("20日低点", _fmt_num(row.get("low_20"), 2), "flat"),
                    _metric_card("距低点", _fmt_num(row.get("low_dist_pct"), 1, "%"), "flat"),
                    _metric_card("近5日涨幅", _fmt_num(row.get("ret_5d"), 1, "%"), "flat"),
                    _metric_card("60日成交额", _fmt_num(amount_yi, 2, "亿"), "flat"),
                    _metric_card("PE", _fmt_num(row.get("pe_ttm"), 1), "flat"),
                    _metric_card("PB", _fmt_num(row.get("pb"), 2), "flat"),
                ]
            st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
            if scheme:
                reason = str(row.get("reason") or "").strip()
                if reason:
                    st.caption(reason)
            st.markdown(_theme_html(profile), unsafe_allow_html=True)
            daily = klines.get(code)
            if daily is None or daily.empty:
                st.caption("没有足够的日线画出 K 线。")
            else:
                _render_kline(daily, row.get("date"))


def _scheme_frame(candidates: pd.DataFrame, scheme: str) -> pd.DataFrame:
    if candidates is None or candidates.empty or "scheme" not in candidates.columns:
        return pd.DataFrame()
    out = candidates[candidates["scheme"] == scheme]
    return out.reset_index(drop=True)


def _render_scheme_block(
    title: str,
    caption: str,
    frame: pd.DataFrame,
    labels: list[tuple[str, str]],
    profiles: dict[str, dict[str, Any]],
    klines: dict[str, pd.DataFrame],
    empty_text: str,
) -> None:
    _section(title)
    st.caption(caption)
    if frame is None or frame.empty:
        st.info(empty_text)
        return
    view = _candidate_view(frame, profiles)
    st.dataframe(_labeled_frame(view, labels), hide_index=True, use_container_width=True)
    st.caption("默认收起。黄色竖线是假设买入日。")
    _render_stock_cards(view, profiles, klines, section=None)


def render_result(result: dict[str, Any]) -> None:
    if int(result.get("n_daily") or 0) == 0 and int(result.get("daily_fetch_failures") or 0) > 0:
        st.error("行情接口失败：新浪和东方财富日线都未取到数据。请检查网络，或稍后重试。")
        return
    if result.get("from_cache"):
        st.caption("行情使用本次会话缓存，只重新计算了买入日名单。")
    candidates = result.get("candidates")
    if result.get("style") == "short":
        trend = _scheme_frame(candidates, "趋势涨")
        pull = _scheme_frame(candidates, "回踩止跌")
        st.caption(f"有效股票 {result['n_daily']} 只。趋势涨 {len(trend)} 只，回踩止跌 {len(pull)} 只。")
        codes: list[str] = []
        if candidates is not None and not candidates.empty:
            codes = [str(code).zfill(6) for code in candidates["code"].tolist()]
        profiles = _load_profiles(
            list(dict.fromkeys(codes)),
            Path(str(result.get("cache_dir") or PROJECT_ROOT / "data_cache")),
        )
        klines = result.get("klines") or {}
        _render_scheme_block(
            "趋势涨",
            "收盘后筛选，供次日早盘买入。收盘偏弱、振幅过大、当天涨完或还在跌的已排除。贴着更早压力、上方空间不足 3% 的已排除。",
            trend,
            TREND_LABELS,
            profiles,
            klines,
            "这一天没有符合趋势涨的股票。阈值在 config.yaml 的 strategy_short。",
        )
        _render_scheme_block(
            "回踩止跌",
            "收盘后筛选，供次日早盘买入。收盘偏弱、振幅过大、当天涨完或还在跌的已排除。贴着更早压力、上方空间不足 3% 的已排除。",
            pull,
            PULL_LABELS,
            profiles,
            klines,
            "这一天没有符合回踩止跌的股票。阈值在 config.yaml 的 strategy_short。",
        )
        return

    st.caption(f"有效股票 {result['n_daily']} 只，符合中线低位企稳 {result['n_candidates']} 只。")

    _section("假设买入日符合标的")
    if candidates is None or candidates.empty:
        st.info("这一天没有符合的股票。策略阈值在 config.yaml。")
    else:
        codes = [str(code).zfill(6) for code in candidates["code"].tolist()]
        profiles = _load_profiles(codes, Path(str(result.get("cache_dir") or PROJECT_ROOT / "data_cache")))
        view = _candidate_view(candidates, profiles)
        st.caption("距低点越接近 0，越贴着近 20 日低点。最近 5 天再创新低的不会出现。")
        st.dataframe(_labeled_frame(view, CANDIDATE_LABELS), hide_index=True, use_container_width=True)
        _render_stock_cards(view, profiles, result.get("klines") or {})


def main() -> None:
    st.set_page_config(page_title="选股筛选", layout="wide", initial_sidebar_state="expanded")
    _inject_css()

    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        st.error(f"找不到配置文件: {config_path}")
        return
    base = load_config(config_path)
    ensure_network(base)

    data_cfg = base.get("data", {}) or {}
    default_asof = base.get("as_of_date") or base.get("end_date") or base.get("start_date")

    with st.sidebar:
        st.markdown('<div class="side-label">控制台</div>', unsafe_allow_html=True)
        st.caption("中线阈值在 strategy，短线阈值在 strategy_short。")

    with st.sidebar.form("screen_form"):
        _side_label("策略")
        style = st.radio(
            "策略",
            options=["mid", "short"],
            format_func=lambda item: "中线低位企稳" if item == "mid" else "短线",
            index=0 if resolve_style(base) == "mid" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        _side_label("买入日")
        as_of_date = st.date_input(
            "假设买入日",
            value=_as_date(default_asof),
            help="不是交易日时，用不超过它的最近一个交易日。",
        )
        max_stocks = st.number_input(
            "股票数上限",
            min_value=1,
            value=int(data_cfg.get("max_stocks", 200)),
            step=50,
            help="全市场扫描很慢，可先用较小数量。",
        )
        submitted = st.form_submit_button("开始筛选", type="primary", use_container_width=True)

    existing = st.session_state.get("result")
    if submitted:
        header_style = style
    elif isinstance(existing, dict) and existing.get("style"):
        header_style = str(existing.get("style"))
    else:
        header_style = style
    _render_header(as_of_date, header_style)
    note = st.session_state.get("network_note") or ""
    if note:
        if st.session_state.get("network_warn"):
            st.warning(note)
        else:
            st.caption(note)

    if submitted:
        form = {
            "style": style,
            "as_of_date": as_of_date,
            "max_stocks": max_stocks,
        }
        errors = validate_form(form)
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            status = st.empty()
            bar = st.progress(0.0)
            stage_box = {"name": None}

            def on_progress(done: int, total: int, stage: str) -> None:
                if stage_box["name"] != stage:
                    stage_box["name"] = stage
                    bar.progress(0.0)
                if total <= 0:
                    status.text(stage)
                    return
                status.text(f"{stage}：{done} / {total}")
                bar.progress(min(done / total, 1.0))

            try:
                st.session_state["result"] = run_screen_job(apply_form(base, form), on_progress)
                bar.progress(1.0)
                done = st.session_state["result"]
                if int(done.get("n_daily") or 0) == 0 and int(done.get("daily_fetch_failures") or 0) > 0:
                    status.error("行情接口失败：新浪和东方财富日线都未取到数据。")
                else:
                    status.success("筛选完成。")
            except Exception as exc:
                status.error(f"筛选失败：{exc}")

    result = st.session_state.get("result")
    if result:
        render_result(result)
    else:
        _render_empty()


main()
