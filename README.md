# 中线埋伏回测程序（PyCharm 一键跑）

把原短线「涨停余温 + 当日大涨」改成 **下跌/冷却埋伏**，持股约 1–2 个月。  
在 PyCharm 里改日期、点运行，即可得到候选股（含盈亏比、基本面、推荐理由）和区间回测摘要。

## 环境要求

- Windows + Python 3.9+
- PyCharm（或任意能运行 `main.py` 的 IDE）

## 安装

在项目根目录 `股票` 下：

```bash
pip install -r requirements.txt
```

PyCharm：`File → Settings → Project → Python Interpreter` 安装 `requirements.txt` 中的包。

## 怎么跑

1. 打开本文件夹作为项目
2. 编辑 [`config.yaml`](config.yaml)：
   - `start_date` / `end_date`：回测与选股区间
   - `mode`：`screen`（只选股）| `backtest`（只回测）| `both`（默认）
   - `data.source`：`akshare`（默认，免 token）| `tushare` | `csv`
   - `data.max_stocks`：首版默认 200，全市场会很慢，可逐步加大
3. 右键运行 [`main.py`](main.py)
4. 看终端摘要，详细结果在 `output/` 的 Excel

网页回测：在项目根目录运行 `streamlit run app.py`，在浏览器里选日期和常用参数后点「开始回测」。结果只在页面上查看，不导出 Excel。页面上没有的策略阈值仍用 `config.yaml`。

## 输出说明

Excel 一般包含：

| Sheet | 内容 |
|--------|------|
| candidates | 期末候选股：盈亏比、止损/止盈、PE/PB、推荐理由 |
| backtest_summary | 胜率、平均盈亏、实现盈亏比、最大回撤、总收益 |
| trades | 每笔买卖明细与退出原因 |
| equity | 资金曲线 |
| all_signals | 区间内全部信号（复盘用） |

## 策略要点（可在 config 调）

- 主板；排除 ST、科创板、创业板
- 冷却埋伏：近 20 日无涨停、缩量、近 60 日已回撤，不追涨
- 盈亏比：止损 -12%；目标按 60 日高点修复比例测算，**R ≥ 1.5** 才入选
- 回测：信号日收盘后入选 → 次日开盘买 → 止损/止盈/到期（20–40 交易日）卖

## 多数据源

| source | 说明 |
|--------|------|
| akshare | 默认免费源，自动缓存到 `data_cache/` |
| tushare | 在 `config.yaml` 填写 `data.tushare_token`，并 `pip install tushare` |
| csv | 离线数据，目录见下 |

### CSV 目录（`data_cache/csv/`）

```text
stock_list.csv          # code,name,circ_mv,is_st
fundamentals.csv        # code,name,pe_ttm,pb,circ_mv,total_mv,revenue_yoy,profit_yoy,is_st
daily/600000.csv        # date,open,high,low,close,volume,amount,turnover,pct_chg
```

## 常见问题

- **很慢**：减小 `data.max_stocks`，或第二次跑会用本地缓存更快  
- **期末无候选**：条件偏严属正常，可略放宽 `drawdown_60d_*` / `ret_60d_*` / `min_rr`  
- **开着 VPN 报 ProxyError / Connection aborted**：程序默认 `data.vpn_split_domestic: true`：保留 VPN 代理，对东方财富/新浪等行情域名 **强制直连**（不经系统代理），东财失败自动回退新浪。若仍失败，在 Clash/VPN 为 `*.eastmoney.com`、`*.sina.com.cn` 设 DIRECT 或开启「绕过中国大陆」  
- **akshare 其它报错**：多为接口变更，可改 `csv`/`tushare`，或隔日重试  
- **仅作研究工具**：不构成投资建议，不自动下单  

## 项目结构

```text
main.py                 # PyCharm 入口
app.py                  # 本地网页入口（streamlit run app.py）
config.yaml             # 日期与策略参数
src/data/               # 多渠道数据适配
src/strategy/           # 过滤、盈亏比、推荐理由
src/backtest/           # 回测与绩效
src/pipeline.py         # 串联流程
output/                 # 结果导出
data_cache/             # 行情缓存
```
