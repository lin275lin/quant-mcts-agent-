"""单股历史因子诊断，打印各因子与未来 5 至 10 日收益的整段 Spearman 相关。"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 直接运行 test 目录里的脚本时，把项目根目录加到 Python 的模块查找路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FACTOR_LABELS, FACTOR_NAMES, LOOKBACK_CALENDAR_DAYS
from tools.fetch_stock_factors import _helpers, _sources, _symbols, login_baostock, logout_baostock


GROUPS = {
    "A": ("MOM20", "MOM60", "MOM120", "REV5"),
    "B": ("VOL20", "VOL60", "DVOL20"),
    "C": ("TURN20", "AMIHUD20", "SIZE"),
    "D": ("INTRA20", "CORR20", "VR"),
    "E": ("RSI14", "AMP20"),
    "F": ("EP", "BP", "SP", "ROE", "REVG"),
}
SNAPSHOT_FACTORS = {"SIZE", "ROE", "REVG"}
STANDARDIZATION_WINDOW = 40
MIN_SAMPLES = 61


def _load_history(symbol: str, start: date, end: date) -> dict:
    """长区间按年读取，复用内部工具的缓存和同一个 BaoStock 会话。"""
    if (end - start).days <= 366:
        return _sources._daily_payload(symbol, start.isoformat(), end.isoformat())

    periods = [
        (max(start, date(year, 1, 1)).isoformat(), min(end, date(year, 12, 31)).isoformat())
        for year in range(start.year, end.year + 1)
    ]
    # 使用现有日线缓存的键。如果各年都命中，就直接读本地缓存。
    cached = [
        _sources._cache_read("daily", symbol, first, last, _sources._ADJUST_FLAG, _sources._DAILY_FIELDS)
        for first, last in periods
    ]
    needs_session = any(payload is None for payload in cached)
    if needs_session:
        login_baostock()
    combined = None
    try:
        for (first, last), payload in zip(periods, cached):
            if payload is None:
                print(f"正在补取 {first} 至 {last} 的日线", flush=True)
                try:
                    payload = _sources._daily_payload(symbol, first, last)
                except RuntimeError as error:
                    # 上市前的年份可以是空的。查询错误要上报，防止误用残缺的历史。
                    if str(error) == f"BaoStock 返回空日线: {symbol}":
                        continue
                    raise
            if combined is None:
                combined = {"fields": payload["fields"], "rows": []}
            combined["rows"].extend(payload["rows"])
    finally:
        if needs_session:
            logout_baostock()
    if combined is None:
        raise RuntimeError(f"BaoStock 返回空日线: {symbol}")
    return combined


def _factor_history(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用现有辅助函数一次计算整段价量和估值因子。"""
    bars, _ = _sources._parse_daily_bars(payload)
    if not bars.index.is_unique:
        raise ValueError("日线有重复交易日期")

    # 现有的分组辅助函数返回逐日序列。这里按日期合并即可复用原公式。
    close = bars["close"]
    returns = close.pct_change()
    price_factors = pd.DataFrame({
        **_helpers._momentum_factors(close),
        **_helpers._volatility_factors(returns),
        **_helpers._liquidity_factors(bars, returns),
        **_helpers._volume_price_factors(bars),
        **_helpers._technical_factors(bars),
    }, index=bars.index)

    # 原始日线同时带逐日估值。每行传给原工具的估值函数，保留有效负值。
    raw = pd.DataFrame(payload["rows"], columns=payload["fields"])
    valuation_rows = raw[_helpers._VALUATION_COLUMNS].to_dict(orient="records")
    valuations = pd.DataFrame(
        [_helpers._valuation_factors(row) for row in valuation_rows],
        index=pd.to_datetime(raw["date"]),
    )
    # 内部工具只提供 SIZE、ROE、REVG 的最新快照，历史列留空。
    factors = price_factors.join(valuations).reindex(columns=FACTOR_NAMES)
    return bars, factors


def _standardize(factors: pd.DataFrame) -> pd.DataFrame:
    """用当天之前的 40 个交易观测标准化，结果限制在 ±3。"""
    values = factors.replace([np.inf, -np.inf], np.nan)
    # shift(1) 先把表向下移一行，rolling 就只会用到今天之前的数据。
    past = values.shift(1).rolling(STANDARDIZATION_WINDOW)
    maximum = past.max()
    constant = maximum.eq(past.min())
    stdev = past.std(ddof=0).mask(constant).replace(0, np.nan)
    z = (values - past.mean()) / stdev
    # 窗口内没有变化，且当天也和过去一样时，按方案给 0。当天变化则仍缺失。
    return z.mask(constant & values.eq(maximum), 0.0).clip(-3, 3)


def _forward_return(bars: pd.DataFrame, horizon: int, next_open: bool) -> pd.Series:
    """在已截好的训练区间内计算收益，末尾未到期的标签自然为空。"""
    prices = bars["open"] if next_open else bars["close"]
    prices = prices.where(prices.gt(0) & np.isfinite(prices))
    # offset=0 是收盘到收盘，offset=1 把进场和离场都顺延到开盘。
    offset = int(next_open)
    return prices.shift(-horizon - offset) / prices.shift(-offset) - 1


def _correlations(factors: pd.DataFrame, future: pd.Series) -> pd.DataFrame:
    """每个因子与同一列后续收益按日期配对，再算一次整段相关。"""
    rows = []
    for name in FACTOR_NAMES:
        pairs = pd.concat([factors[name], future], axis=1)
        pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
        count = len(pairs)
        ic = np.nan
        pvalue = np.nan
        if name in SNAPSHOT_FACTORS and count == 0:
            status = "缺少历史数据"
        elif count < MIN_SAMPLES:
            status = "样本不足"
        elif (pairs.nunique() < 2).any():
            status = "因子或收益为常数"
        else:
            # Spearman 会分别给同一只股票的历史因子和收益排名，再按原日期计算相关。
            result = spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1])
            ic = float(result.statistic)
            pvalue = float(result.pvalue)
            status = ""
        rows.append((name, ic, pvalue, count, status))
    return pd.DataFrame(rows, columns=["factor", "ic", "pvalue", "samples", "status"]).set_index("factor")


def _group_weights(ics: pd.Series, threshold: float) -> pd.Series:
    """过滤低于阈值的 IC，再在各组内按绝对值总和归一化。"""
    usable = ics.where(ics.abs() >= threshold, 0.0)
    weights = pd.Series(0.0, index=ics.index)
    for names in GROUPS.values():
        group_ics = usable.loc[list(names)]
        total = group_ics.abs().sum()
        if total > 0:
            weights.loc[list(names)] = group_ics / total
    return weights


def _print_results(results: pd.DataFrame, horizon: int, threshold: float) -> None:
    """把相关度和用于评分的权重分开打印，便于核对弱信号和缺失。"""
    weights = _group_weights(results["ic"], threshold)
    print(f"\n未来 {horizon} 日收益")
    print("组  指标代码      Spearman IC    样本数      普通 p 值   组内权重  中文指标 / 诊断")
    for group, names in GROUPS.items():
        for name in names:
            ic, pvalue, count, status = results.loc[name]
            display = f"{ic:+.4f}" if pd.notna(ic) else "NULL"
            p_display = f"{pvalue:.3g}" if pd.notna(pvalue) else "NULL"
            if not status:
                if abs(ic) < threshold:
                    status = "低于筛选阈值"
                elif ic == 0:
                    status = "零相关"
                else:
                    status = "正相关" if ic > 0 else "负相关"
            print(f"{group}   {name:<9} {display:>12} {count:>9} {p_display:>14} "
                  f"{weights[name]:>+10.4f}  {FACTOR_LABELS[name]} / {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="601868", help="股票代码或精确名称，默认 601868")
    parser.add_argument("--start-date", type=date.fromisoformat, help="训练开始日，默认截止日前五年")
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today(), help="训练截止日")
    parser.add_argument("--horizons", type=int, nargs="+", choices=range(5, 11), default=[5, 10],
                        help="要检查的持有期，默认 5 10")
    parser.add_argument("--threshold", type=float, default=0.02, help="组内过滤用的 |IC| 阈值")
    parser.add_argument("--next-open", action="store_true", help="用次日开盘开始的收益，默认收盘到收盘")
    parser.add_argument("--rolling-z", action="store_true",
                        help=f"用前 {STANDARDIZATION_WINDOW} 个交易日标准化后的因子，默认用因子原值")
    args = parser.parse_args()

    end = args.end_date
    start = args.start_date or (pd.Timestamp(end) - pd.DateOffset(years=5)).date()
    if start >= end:
        parser.error("开始日必须早于截止日")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("IC 阈值需在 0 到 1 之间")

    # 多取开始日前的一段历史，给 MOM120 和可选的滚动标准化预热。
    warmup_start = start - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    try:
        symbol = _symbols._resolve(args.symbol)
        print(f"正在读取 {symbol} 的历史日线（{warmup_start} 至 {end}）", flush=True)
        payload = _load_history(symbol, warmup_start, end)
        bars, raw = _factor_history(payload)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"读取历史日线失败：{error}\n")

    train_bars = bars.loc[start.isoformat():end.isoformat()]
    if train_bars.empty:
        parser.exit(1, "指定的训练日期内没有日线\n")
    # Spearman 自行排名。默认直接用因子值，滚动标准化留作另一种信号的对照。
    factors = _standardize(raw) if args.rolling_z else raw
    factors = factors.reindex(train_bars.index)
    factor_basis = f"前 {STANDARDIZATION_WINDOW} 日滚动标准化" if args.rolling_z else "原值"
    basis = "次日开盘到持有期末开盘" if args.next_open else "当日收盘到持有期末收盘"
    print(f"股票 {symbol}  实际训练日线 {train_bars.index[0].date()} 至 "
          f"{train_bars.index[-1].date()}，共 {len(train_bars)} 行")
    print(f"收益口径 {basis}  因子口径 {factor_basis}  "
          f"最低 {MIN_SAMPLES} 对样本  组内筛选 |IC| ≥ {args.threshold:g}")
    print("SIZE、ROE、REVG 在现有工具里为快照，历史诊断显示 NULL。")
    print("相邻的后续收益窗口有重叠，普通 p 值仅供参考。")

    has_result = False
    # dict.fromkeys 保留参数原有顺序，同时去掉重复的持有期。
    for horizon in dict.fromkeys(args.horizons):
        future = _forward_return(train_bars, horizon, args.next_open)
        results = _correlations(factors, future)
        _print_results(results, horizon, args.threshold)
        has_result = has_result or results["ic"].notna().any()
    return 0 if has_result else 1


if __name__ == "__main__":
    raise SystemExit(main())
