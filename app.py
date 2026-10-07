"""中线埋伏回测的本地网页入口。

启动：streamlit run app.py
结果只在页面上查看，不导出 Excel。命令行 main.py 不受影响。
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
from src.backtest.engine import run_backtest
from src.backtest.metrics import summarize_trades, trades_to_frame
from src.data.base import get_provider
from src.data.concepts import fetch_stock_profile
from src.data.network_util import apply_vpn_friendly_split, install_akshare_direct_patch, use_sina_quote_source
from src.pipeline import (
    build_fund_map,
    collect_stock_data,
    first_signals,
    generate_signals,
    screen_period,
)

MODE_LABELS = {
    "选股 + 回测": "both",
    "只选股": "screen",
    "只回测": "backtest",
}
MODE_BY_VALUE = {value: label for label, value in MODE_LABELS.items()}

CANDIDATE_LABELS = [
    ("date", "日期"),
    ("code", "代码"),
    ("name", "名称"),
    ("signal_close", "收盘价"),
    ("drawdown_250", "一年回撤(%)"),
    ("range_pos_pct", "区间位置(%)"),
    ("amount_yi", "日均成交额(亿)"),
    ("pe_ttm", "PE"),
    ("pb", "PB"),
    ("circ_mv", "流通市值(亿)"),
    ("industry", "行业"),
]
TRADE_LABELS = [
    ("code", "代码"),
    ("name", "名称"),
    ("entry_date", "买入日"),
    ("exit_date", "卖出日"),
    ("entry_price", "买入价"),
    ("exit_price", "卖出价"),
    ("shares", "股数"),
    ("ret_pct", "收益率(%)"),
    ("pnl", "盈亏"),
    ("reason", "退出原因"),
    ("realized_rr", "实现盈亏比"),
]
SUMMARY_LABELS = {
    "trades": "交易笔数",
    "win_rate": "胜率 (%)",
    "avg_win_pct": "平均盈利 (%)",
    "avg_loss_pct": "平均亏损 (%)",
    "avg_realized_rr": "实现盈亏比",
    "total_pnl": "总盈亏",
    "total_return_pct": "总收益率 (%)",
    "max_drawdown_pct": "最大回撤 (%)",
    "final_equity": "期末权益",
}


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
    if form["start_date"] >= form["end_date"]:
        errors.append("开始日期必须早于结束日期。")
    if form["hold_min_days"] > form["hold_max_days"]:
        errors.append("最短持有天数不能大于最长持有天数。")
    if form["initial_cash"] <= 0:
        errors.append("初始资金必须为正数。")
    if form["max_positions"] <= 0 or form["max_stocks"] <= 0:
        errors.append("最大持仓和股票数上限必须为正数。")
    if form["hold_min_days"] <= 0 or form["hold_max_days"] <= 0:
        errors.append("持有天数必须为正数。")
    if form["stop_loss_pct"] <= 0 or form["take_profit_pct"] <= 0:
        errors.append("止损和止盈上限必须为正数。")
    if form["drawdown_min"] < 0:
        errors.append("回撤至少不能为负数。")
    if not 0 <= form["range_pos_pct"] <= 100:
        errors.append("低位位置上限应在 0 到 100 之间。")
    if form["amount_min_yi"] < 0:
        errors.append("日均成交额下限不能为负数。")
    return errors


def apply_form(base: dict[str, Any], form: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["start_date"] = form["start_date"].strftime("%Y-%m-%d")
    cfg["end_date"] = form["end_date"].strftime("%Y-%m-%d")
    cfg["mode"] = MODE_LABELS[form["mode_label"]]
    cfg.setdefault("data", {})
    cfg.setdefault("strategy", {})
    cfg.setdefault("backtest", {})
    cfg["data"]["max_stocks"] = int(form["max_stocks"])
    strategy = cfg["strategy"]
    strategy["max_positions"] = int(form["max_positions"])
    strategy["hold_min_days"] = int(form["hold_min_days"])
    strategy["hold_max_days"] = int(form["hold_max_days"])
    strategy["stop_loss_pct"] = float(form["stop_loss_pct"]) / 100.0
    strategy["take_profit_cap_pct"] = float(form["take_profit_pct"]) / 100.0
    strategy["drawdown_min"] = float(form["drawdown_min"])
    strategy["range_pos_max"] = float(form["range_pos_pct"]) / 100.0
    strategy["amount_median_min_yi"] = float(form["amount_min_yi"])
    cfg["backtest"]["initial_cash"] = float(form["initial_cash"])
    return cfg


def _fill_circ_mv(circ_mv_map: dict[str, float], fund_map: dict[str, dict[str, Any]]) -> None:
    for code, fund in fund_map.items():
        current = circ_mv_map.get(code)
        missing = code not in circ_mv_map or pd.isna(current)
        if missing and pd.notna(fund.get("circ_mv")):
            circ_mv_map[code] = float(fund["circ_mv"])


def run_backtest_job(config: dict[str, Any], on_progress) -> dict[str, Any]:
    start_date = str(config["start_date"])
    end_date = str(config["end_date"])
    mode = str(config.get("mode", "both")).lower()
    strategy = config.get("strategy", {})
    backtest_cfg = config.get("backtest", {})
    data_cfg = config.get("data", {}) or {}
    cache_key = (
        start_date,
        end_date,
        str(data_cfg.get("source", "akshare")).lower(),
        int(data_cfg.get("max_stocks", 0) or 0),
        "sina" if use_sina_quote_source() else "eastmoney",
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
            provider, universe, start_date, end_date, strategy, progress=on_progress
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

    signals = generate_signals(
        daily_map,
        circ_mv_map,
        name_map,
        start_date,
        end_date,
        strategy,
        fund_map,
        progress=on_progress,
    )
    candidates = screen_period(signals)
    summary: dict[str, Any] = {}
    trades = pd.DataFrame()
    equity = pd.DataFrame()
    ran_backtest = False
    if mode in ("backtest", "both") and signals is not None and not signals.empty:
        bt_signals = first_signals(signals)
        result = run_backtest(bt_signals, daily_map, strategy, backtest_cfg, name_map)
        summary = summarize_trades(result, float(backtest_cfg.get("initial_cash", 1_000_000)))
        trades = trades_to_frame(result.trades)
        equity = result.equity_curve
        ran_backtest = True

    n_signals = 0 if signals is None or signals.empty else len(signals)
    fetch_failures = 0 if from_cache else int(getattr(provider, "daily_fetch_failures", 0) or 0)
    return {
        "mode": mode,
        "from_cache": from_cache,
        "n_daily": len(daily_map),
        "n_signals": n_signals,
        "daily_fetch_failures": fetch_failures,
        "candidates": candidates,
        "klines": _candidate_klines(daily_map, candidates),
        "cache_dir": str(PROJECT_ROOT / str(data_cfg.get("cache_dir", "data_cache"))),
        "summary": summary,
        "trades": trades,
        "equity": equity,
        "ran_backtest": ran_backtest,
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


def _render_header(start_date: date, end_date: date, mode_label: str) -> None:
    st.markdown(
        f"""
        <div class="term-head">
          <div>
            <div class="term-kicker">A-SHARE RESEARCH</div>
            <div class="term-title">中线低位 <span>筛选</span></div>
          </div>
          <div class="term-pills">
            <span class="pill">{start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}</span>
            <span class="pill">{mode_label}</span>
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
          <p>在左侧设定区间和低位条件，然后开始筛选。结果只在本页查看，市值下限和估值上限仍使用 config.yaml。</p>
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


def _candidate_view(candidates: pd.DataFrame, profiles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    view = candidates.copy()
    view["code"] = view["code"].astype(str).str.zfill(6)
    view["range_pos_pct"] = pd.to_numeric(view.get("range_pos"), errors="coerce") * 100
    view["amount_yi"] = pd.to_numeric(view.get("amount_median"), errors="coerce") / 1e8
    view["industry"] = view["code"].map(
        lambda code: (profiles.get(code) or {}).get("industry") or "—"
    )
    for col, digits in (
        ("signal_close", 2),
        ("drawdown_250", 1),
        ("range_pos_pct", 1),
        ("amount_yi", 2),
        ("pe_ttm", 1),
        ("pb", 2),
        ("circ_mv", 1),
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
) -> None:
    _section("个股详情")
    st.caption("默认收起。黄色竖线是这只股票在筛选区间里最后一次符合的交易日。")
    for _, row in candidates.iterrows():
        code = str(row["code"]).zfill(6)
        name = str(row.get("name") or code)
        profile = profiles.get(code) or {}
        title = f"{code}  {name}    {_fmt_num(row.get('signal_close'), 2)} 元"
        with st.expander(title):
            pos = row.get("range_pos")
            try:
                pos_pct = float(pos) * 100
            except (TypeError, ValueError):
                pos_pct = float("nan")
            amount = row.get("amount_median")
            try:
                amount_yi = float(amount) / 1e8
            except (TypeError, ValueError):
                amount_yi = float("nan")
            cards = "".join(
                [
                    _metric_card("现价", _fmt_num(row.get("signal_close"), 2), "flat"),
                    _metric_card("一年回撤", _fmt_num(row.get("drawdown_250"), 1, "%"), "down"),
                    _metric_card("区间位置", _fmt_num(pos_pct, 1, "%"), "flat"),
                    _metric_card("PE", _fmt_num(row.get("pe_ttm"), 1), "flat"),
                    _metric_card("PB", _fmt_num(row.get("pb"), 2), "flat"),
                    _metric_card("流通市值", _fmt_num(row.get("circ_mv"), 1, "亿"), "flat"),
                    _metric_card("120日成交额", _fmt_num(amount_yi, 2, "亿"), "flat"),
                    _metric_card("符合天数", _fmt_num(row.get("hit_days"), 0, "天"), "flat"),
                ]
            )
            st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)
            st.markdown(_theme_html(profile), unsafe_allow_html=True)
            daily = klines.get(code)
            if daily is None or daily.empty:
                st.caption("没有足够的日线画出 K 线。")
            else:
                _render_kline(daily, row.get("date"))


def render_result(result: dict[str, Any]) -> None:
    if int(result.get("n_daily") or 0) == 0 and int(result.get("daily_fetch_failures") or 0) > 0:
        st.error("行情接口失败：新浪和东方财富日线都未取到数据。请检查网络，或稍后重试。")
        return
    mode = result["mode"]
    if result.get("from_cache"):
        st.caption("行情使用本次会话缓存，只重新计算了信号和回测。")
    st.caption(f"有效股票 {result['n_daily']} 只，区间信号 {result['n_signals']} 条。")

    show_candidates = mode in ("screen", "both")
    show_backtest = mode in ("backtest", "both")

    if show_backtest:
        _section("回测摘要")
        if not result.get("ran_backtest"):
            st.info("区间内没有信号，未进行回测。")
        else:
            summary = result.get("summary") or {}
            ret = float(summary.get("total_return_pct", 0) or 0)
            win = float(summary.get("win_rate", 0) or 0)
            dd = float(summary.get("max_drawdown_pct", 0) or 0)
            sign = "+" if ret > 0 else ""
            cards = "".join(
                [
                    _metric_card("总收益", f"{sign}{ret}%", _tone(ret)),
                    _metric_card("胜率", f"{win}%", "up" if win >= 50 else "down"),
                    _metric_card("最大回撤", f"{dd}%", "down" if dd < 0 else "flat"),
                    _metric_card("期末权益", f"{float(summary.get('final_equity', 0) or 0):,.2f}", "flat"),
                ]
            )
            st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)
            extra = {SUMMARY_LABELS.get(k, k): v for k, v in summary.items()}
            st.dataframe(pd.DataFrame([extra]), hide_index=True, use_container_width=True)
            equity = result.get("equity")
            if equity is not None and not equity.empty and "equity" in equity.columns:
                _section("资金曲线")
                _render_equity(equity)
            trades = result.get("trades")
            _section("交易明细")
            if trades is None or trades.empty:
                st.info("没有成交记录。")
            else:
                st.dataframe(_labeled_frame(trades, TRADE_LABELS), hide_index=True, use_container_width=True)

    if show_candidates:
        _section("区间符合标的")
        candidates = result.get("candidates")
        if candidates is None or candidates.empty:
            st.info("区间内没有符合的股票。可放宽回撤、低位位置或日均成交额，或增大股票数上限。")
        else:
            codes = [str(code).zfill(6) for code in candidates["code"].tolist()]
            profiles = _load_profiles(codes, Path(str(result.get("cache_dir") or PROJECT_ROOT / "data_cache")))
            view = _candidate_view(candidates, profiles)
            st.caption("区间位置越低，越贴近这一年的低点。行业来自入选后补拉的题材数据。")
            st.dataframe(_labeled_frame(view, CANDIDATE_LABELS), hide_index=True, use_container_width=True)
            _render_stock_cards(view, profiles, result.get("klines") or {})


def main() -> None:
    st.set_page_config(page_title="中线低位筛选", layout="wide", initial_sidebar_state="expanded")
    _inject_css()

    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        st.error(f"找不到配置文件: {config_path}")
        return
    base = load_config(config_path)
    ensure_network(base)

    strategy = base.get("strategy", {}) or {}
    backtest_cfg = base.get("backtest", {}) or {}
    data_cfg = base.get("data", {}) or {}
    default_mode = MODE_BY_VALUE.get(str(base.get("mode", "both")).lower(), "选股 + 回测")

    with st.sidebar:
        st.markdown('<div class="side-label">控制台</div>', unsafe_allow_html=True)
        st.caption("市值下限、估值上限仍使用 config.yaml。")

    with st.sidebar.form("backtest_form"):
        _side_label("区间")
        start_date = st.date_input("开始日期", value=_as_date(base.get("start_date")))
        end_date = st.date_input("结束日期", value=_as_date(base.get("end_date")))
        mode_label = st.selectbox(
            "模式",
            list(MODE_LABELS.keys()),
            index=list(MODE_LABELS.keys()).index(default_mode),
        )

        _side_label("资金与持仓")
        initial_cash = st.number_input(
            "初始资金",
            min_value=1.0,
            value=float(backtest_cfg.get("initial_cash", 1_000_000)),
            step=100_000.0,
        )
        max_positions = st.number_input(
            "最大持仓",
            min_value=1,
            value=int(strategy.get("max_positions", 5)),
            step=1,
        )
        hold_c1, hold_c2 = st.columns(2)
        hold_min_days = hold_c1.number_input(
            "最短持有天数",
            min_value=1,
            value=int(strategy.get("hold_min_days", 20)),
            step=1,
            help="只用于回测里的持有区间，不决定能不能入选。",
        )
        hold_max_days = hold_c2.number_input(
            "最长持有天数",
            min_value=1,
            value=int(strategy.get("hold_max_days", 40)),
            step=1,
            help="持有达到该交易日数后，按收盘价到期卖出。",
        )

        _side_label("风控")
        risk_c1, risk_c2 = st.columns(2)
        stop_loss_pct = risk_c1.number_input(
            "止损 (%)",
            min_value=0.1,
            value=round(float(strategy.get("stop_loss_pct", 0.12)) * 100, 2),
            step=0.5,
            help="填写 12 表示下跌 12% 止损。",
        )
        take_profit_pct = risk_c2.number_input(
            "止盈上限 (%)",
            min_value=0.1,
            value=round(float(strategy.get("take_profit_cap_pct", 0.25)) * 100, 2),
            step=0.5,
            help="填写 25 表示目标涨幅不超过 25%。",
        )
        max_stocks = st.number_input(
            "股票数上限",
            min_value=1,
            value=int(data_cfg.get("max_stocks", 200)),
            step=50,
            help="全市场扫描很慢，可先用较小数量。",
        )

        _side_label("低位条件")
        drawdown_min = st.number_input(
            "回撤至少（%）",
            min_value=0.0,
            value=float(strategy.get("drawdown_min", 30)),
            step=1.0,
            help="相对近一年最高价，至少跌这么多才算处在低位。",
        )
        range_pos_pct = st.number_input(
            "低位位置上限（%）",
            min_value=0.0,
            max_value=100.0,
            value=round(float(strategy.get("range_pos_max", 0.35)) * 100, 1),
            step=1.0,
            help="35 表示收盘要落在近一年高低区间里、从低点算起的 35% 以内。",
        )
        amount_min_yi = st.number_input(
            "日均成交额下限（亿）",
            min_value=0.0,
            value=float(strategy.get("amount_median_min_yi", 0.5)),
            step=0.1,
            help="近 120 个交易日成交额中位数，用来表示以前有没有人交易。",
        )
        submitted = st.form_submit_button("开始筛选", type="primary", use_container_width=True)

    _render_header(start_date, end_date, mode_label)
    note = st.session_state.get("network_note") or ""
    if note:
        if st.session_state.get("network_warn"):
            st.warning(note)
        else:
            st.caption(note)

    if submitted:
        form = {
            "start_date": start_date,
            "end_date": end_date,
            "mode_label": mode_label,
            "initial_cash": initial_cash,
            "max_positions": max_positions,
            "hold_min_days": hold_min_days,
            "hold_max_days": hold_max_days,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "max_stocks": max_stocks,
            "drawdown_min": drawdown_min,
            "range_pos_pct": range_pos_pct,
            "amount_min_yi": amount_min_yi,
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
                st.session_state["result"] = run_backtest_job(apply_form(base, form), on_progress)
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
