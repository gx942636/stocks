"""中线埋伏回测的本地网页入口。

启动：streamlit run app.py
结果只在页面上查看，不导出 Excel。命令行 main.py 不受影响。
"""
from __future__ import annotations

import copy
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
from src.data.network_util import apply_vpn_friendly_split, install_akshare_direct_patch, use_sina_quote_source
from src.pipeline import (
    build_fund_map,
    collect_stock_data,
    generate_signals,
    screen_on_date,
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
    ("signal_close", "信号收盘"),
    ("pct_chg", "涨跌幅"),
    ("turnover", "换手率"),
    ("drawdown_60d", "60日回撤"),
    ("ret_60d", "60日涨跌"),
    ("rr", "盈亏比"),
    ("stop_price", "止损价"),
    ("target_price", "目标价"),
    ("pe_ttm", "PE"),
    ("pb", "PB"),
    ("reason", "推荐理由"),
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
    if form["price_min"] >= form["price_max"]:
        errors.append("价格下限必须小于价格上限。")
    if form["circ_mv_min"] >= form["circ_mv_max"]:
        errors.append("流通市值下限必须小于上限。")
    if form["hold_min_days"] > form["hold_max_days"]:
        errors.append("最短持有天数不能大于最长持有天数。")
    if form["initial_cash"] <= 0:
        errors.append("初始资金必须为正数。")
    if form["max_positions"] <= 0 or form["max_stocks"] <= 0:
        errors.append("最大持仓和股票数上限必须为正数。")
    if form["hold_min_days"] <= 0 or form["hold_max_days"] <= 0:
        errors.append("持有天数必须为正数。")
    if form["stop_loss_pct"] <= 0 or form["take_profit_pct"] <= 0 or form["min_rr"] <= 0:
        errors.append("止损、止盈上限和最小盈亏比必须为正数。")
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
    strategy["price_min"] = float(form["price_min"])
    strategy["price_max"] = float(form["price_max"])
    strategy["circ_mv_min_yi"] = float(form["circ_mv_min"])
    strategy["circ_mv_max_yi"] = float(form["circ_mv_max"])
    strategy["max_positions"] = int(form["max_positions"])
    strategy["hold_min_days"] = int(form["hold_min_days"])
    strategy["hold_max_days"] = int(form["hold_max_days"])
    strategy["stop_loss_pct"] = float(form["stop_loss_pct"]) / 100.0
    strategy["take_profit_cap_pct"] = float(form["take_profit_pct"]) / 100.0
    strategy["min_rr"] = float(form["min_rr"])
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
    candidates = screen_on_date(signals, asof=end_date)
    summary: dict[str, Any] = {}
    trades = pd.DataFrame()
    equity = pd.DataFrame()
    ran_backtest = False
    if mode in ("backtest", "both") and signals is not None and not signals.empty:
        bt_signals = (
            signals.sort_values("rr", ascending=False)
            .drop_duplicates(["date", "code"])
            .reset_index(drop=True)
        )
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
        "summary": summary,
        "trades": trades,
        "equity": equity,
        "ran_backtest": ran_backtest,
    }


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
            <div class="term-title">中线埋伏 <span>回测终端</span></div>
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
          <p>在左侧设定区间与风控，然后开始回测。结果只在本页查看，其余策略阈值仍使用 config.yaml。</p>
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
        _section("期末候选股")
        candidates = result.get("candidates")
        if candidates is None or candidates.empty:
            st.info("期末无候选股。可放宽阈值，或增大股票数上限。")
        else:
            st.dataframe(_labeled_frame(candidates, CANDIDATE_LABELS), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="中线埋伏回测", layout="wide", initial_sidebar_state="expanded")
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
        st.caption("其余策略阈值仍使用 config.yaml。")

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
            help="用于推荐理由中的预计持有天数，不决定卖出时点。",
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
        min_rr = st.number_input(
            "最小盈亏比",
            min_value=0.1,
            value=round(float(strategy.get("min_rr", 1.5)), 2),
            step=0.1,
        )
        max_stocks = st.number_input(
            "股票数上限",
            min_value=1,
            value=int(data_cfg.get("max_stocks", 200)),
            step=50,
            help="全市场扫描很慢，可先用较小数量。",
        )

        _side_label("价格与市值")
        price_c1, price_c2 = st.columns(2)
        price_min = price_c1.number_input("价格下限", min_value=0.0, value=float(strategy.get("price_min", 10)), step=1.0)
        price_max = price_c2.number_input("价格上限", min_value=0.0, value=float(strategy.get("price_max", 50)), step=1.0)
        mv_c1, mv_c2 = st.columns(2)
        circ_mv_min = mv_c1.number_input(
            "流通市值下限（亿）",
            min_value=0.0,
            value=float(strategy.get("circ_mv_min_yi", 50)),
            step=10.0,
        )
        circ_mv_max = mv_c2.number_input(
            "流通市值上限（亿）",
            min_value=0.0,
            value=float(strategy.get("circ_mv_max_yi", 300)),
            step=10.0,
        )
        submitted = st.form_submit_button("开始回测", type="primary", use_container_width=True)

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
            "min_rr": min_rr,
            "max_stocks": max_stocks,
            "price_min": price_min,
            "price_max": price_max,
            "circ_mv_min": circ_mv_min,
            "circ_mv_max": circ_mv_max,
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
                    status.success("回测完成。")
            except Exception as exc:
                status.error(f"回测失败：{exc}")

    result = st.session_state.get("result")
    if result:
        render_result(result)
    else:
        _render_empty()


main()
