"""
中线低位企稳 / 短线选股入口。

PyCharm 使用：
1. 打开本项目文件夹
2. 安装依赖: pip install -r requirements.txt
3. 修改 config.yaml 中的 as_of_date，style 用 mid 或 short
4. 右键运行本文件
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base import get_provider
from src.data.network_util import apply_vpn_friendly_split, install_akshare_direct_patch
from src.pipeline import run_pipeline
from src.strategy.style import resolve_style


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    config = load_config(config_path)
    style = resolve_style(config)
    title = "短线两种方案" if style == "short" else "中线低位企稳"
    print("=" * 60)
    print(title)
    print(f"假设买入日: {config.get('as_of_date')}")
    print(f"数据源: {config.get('data', {}).get('source')}")
    print("=" * 60)

    data_cfg = config.get("data", {}) or {}
    split_enabled = bool(data_cfg.get("vpn_split_domestic", True))
    if str(data_cfg.get("source", "akshare")).lower() == "akshare" and split_enabled:
        split_info = apply_vpn_friendly_split(True)
        install_akshare_direct_patch()
        print(
            "网络分流: 已保留 VPN/系统代理；国内行情强制直连；"
            f"clist探测={'OK' if split_info.get('probe_ok') else '失败'}"
        )
        print("  行情主源: 新浪，失败时再试东方财富。")
        if split_info.get("probe_detail"):
            print(f"  探测详情: {split_info.get('probe_detail')}")
        if split_info.get("probe_ok") is False:
            print("  提示: 东财探测失败，备用源可能不可用。")
    elif not split_enabled:
        print("网络分流: 已关闭（data.vpn_split_domestic=false）")

    provider = get_provider(config, PROJECT_ROOT)
    result = run_pipeline(config, PROJECT_ROOT, provider)

    candidates = result.get("candidates")
    if candidates is not None and not candidates.empty:
        if style == "short":
            groups = [
                (
                    "趋势涨",
                    ["date", "code", "name", "signal_close", "ret_20d", "drawdown_10", "pct_chg", "amount_ratio", "close_pos", "resist_price", "resist_room", "reason"],
                ),
                (
                    "回踩止跌",
                    ["date", "code", "name", "signal_close", "drawdown_10", "low_8", "low_dist_8", "ret_3d", "amount_ratio", "close_pos", "resist_price", "resist_room", "reason"],
                ),
            ]
            for scheme, wanted in groups:
                part = candidates[candidates["scheme"] == scheme] if "scheme" in candidates.columns else candidates.iloc[0:0]
                print(f"\n{scheme}（{len(part)} 只，显示前 20）：")
                if part.empty:
                    print("无")
                    continue
                cols = [col for col in wanted if col in part.columns]
                print(part[cols].head(20).to_string(index=False))
        else:
            cols = [
                c
                for c in [
                    "date",
                    "code",
                    "name",
                    "signal_close",
                    "drawdown_60",
                    "low_20",
                    "low_dist_pct",
                    "ret_5d",
                    "amount_median",
                    "pe_ttm",
                    "pb",
                    "reason",
                ]
                if c in candidates.columns
            ]
            print("\n假设买入日符合标的（前 20）：")
            print(candidates[cols].head(20).to_string(index=False))
    elif style == "short":
        print("\n这一天没有符合短线方案的股票。")
    else:
        print("\n这一天没有符合中线低位企稳的股票。")

    print(f"\n完成。Excel: {result.get('output')}")


if __name__ == "__main__":
    main()
