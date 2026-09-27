"""
中线埋伏选股 / 回测入口。

PyCharm 使用：
1. 打开本项目文件夹
2. 安装依赖: pip install -r requirements.txt
3. 修改 config.yaml 中的 start_date / end_date
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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    config = load_config(config_path)
    print("=" * 60)
    print("中线埋伏回测程序")
    print(f"区间: {config.get('start_date')} ~ {config.get('end_date')}")
    print(f"模式: {config.get('mode')}")
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
        cols = [
            c
            for c in [
                "date",
                "code",
                "name",
                "signal_close",
                "drawdown_60d",
                "rr",
                "stop_price",
                "target_price",
                "pe_ttm",
                "pb",
                "reason",
            ]
            if c in candidates.columns
        ]
        print("\n期末候选股（前 20）：")
        print(candidates[cols].head(20).to_string(index=False))
    else:
        print("\n期末无候选股。可放宽 config.yaml 中 strategy 阈值，或扩大 data.max_stocks。")

    print(f"\n完成。Excel: {result.get('output')}")


if __name__ == "__main__":
    main()
