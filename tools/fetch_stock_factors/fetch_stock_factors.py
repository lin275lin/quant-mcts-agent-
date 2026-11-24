"""单股因子工具的公开入口，负责取数据、计算和组装结果。"""
from __future__ import annotations
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any
# 同时支持在项目中导入和从任意工作目录直接运行此文件。
if not __package__:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.fetch_stock_factors import _helpers, _sources, _symbols
else:
    from . import _helpers, _sources, _symbols
from config import FACTOR_LABELS, FACTOR_NAMES
def fetch_stock_factor_vector(
    symbol: str,
    data_end_date: str | None = None,
    *,
    include_daily_history: bool = False,
) -> dict[str, Any]:
    """返回单股最近可用日的 20 个原始因子。

    接受股票代码、带交易所的代码或精确股票名称。截止日用于日线和随日线
    返回的估值。总市值和财务取各自数据源的最新快照。metadata 提供核对所需的
    日期和原始值，include_daily_history=True 时额外返回整段原始日线。
    """
    # 1. 把截止日字符串转为日期，未填时使用今天。
    end = date.fromisoformat(data_end_date) if data_end_date else date.today()
    # 2. 将输入统一成 600519.SH 这种写法。市值接口只需要点号前的部分。
    standard_symbol = _symbols._resolve(symbol)
    code, _ = standard_symbol.split(".")

    # 3. submit() 让市值和财务在后台开始查询，当前线程同时取日线与估值。
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stock-factors") as workers:
        market_cap_future = workers.submit(_sources._fetch_market_cap, code)
        financial_future = workers.submit(_sources._fetch_financial_indicators, standard_symbol)
        bars, daily, daily_history = _sources._fetch_daily_bars(standard_symbol, end)

        # 4. 用日线表计算价量因子，再合并估值、市值和财务。result() 等待查询结果。
        price_factors = _helpers._compute_price_factors(bars)
        factors = price_factors.to_dict()
        factors.update(_helpers._valuation_factors(daily["latest"]))
        market_cap = market_cap_future.result()
        financial = financial_future.result()
        factors["SIZE"] = _helpers._size_factor(market_cap["value_100m_cny"])
        factors.update(_helpers._financial_factors(financial["ROEJQ"], financial["TOTALOPERATEREVETZ"]))

    # 5. 字典和列表统一按配置顺序输出，缺失字段保留 None。
    trading_day = price_factors.name.date()
    vector = {name: factors.get(name) for name in FACTOR_NAMES}
    result = {
        "success": True,
        "symbol": standard_symbol,
        "data_date": trading_day.isoformat(),
        "vector": vector,
        "vector_values": list(vector.values()),
        "factor_names": FACTOR_NAMES.copy(),
        "metadata": {"daily": daily, "market_cap": market_cap, "financial": financial},
    }
    # 日线已经查过，选择显示整段历史时直接交回接口原有的字段和行。
    if include_daily_history:
        result["daily_history"] = daily_history
    return result


def _print_source_metadata(result: dict[str, Any]) -> None:
    """按数据源打印原始指标，方便拿着日期、单位和字段名去对照。"""

    def show(value: Any) -> str:
        # 只能用 is None 判断缺失，否则 0 也会被当成空值。
        return "NULL" if value is None else str(value)

    metadata = result["metadata"]
    daily = metadata["daily"]
    latest = daily["latest"]
    print(f"\n核对用原始数据（{result['symbol']}）")
    print(
        f"  {daily['source']} 日线（{daily['adjustment']}）"
        f"  请求 {daily['requested_start']} 至 {daily['requested_end']}"
    )
    print(f"    实际返回 {daily['first_date']} 至 {latest['date']}，共 {daily['row_count']} 条")
    print(f"    价量因子日期 {result['data_date']}，下列行情和估值日期 {latest['date']}")
    for field, label, unit in (
        ("open", "开盘价", "元"), ("high", "最高价", "元"),
        ("low", "最低价", "元"), ("close", "收盘价", "元"),
        ("volume", "成交量", "股"), ("amount", "成交额", "元"),
        ("turn", "换手率", "%"),
        ("peTTM", "滚动市盈率", "倍"), ("pbMRQ", "市净率（最近报告期）", "倍"),
        ("psTTM", "滚动市销率", "倍"),
    ):
        print(f"    {label}（{field}，单位 {unit}） = {show(latest[field])}")

    market_cap = metadata["market_cap"]
    print(f"  {market_cap['source']}  市值日期 {show(market_cap['date'])}")
    print(f"    总市值（value，单位 亿元） = {show(market_cap['value_100m_cny'])}")

    financial = metadata["financial"]
    print(
        f"  {financial['source']}  报告期 {show(financial['report_date'])}"
        f"  公告日 {show(financial['announcement_date'])}"
    )
    print(f"    加权净资产收益率（ROEJQ，单位 %） = {show(financial['ROEJQ'])}")
    print(f"    营业收入同比（TOTALOPERATEREVETZ，单位 %） = {show(financial['TOTALOPERATEREVETZ'])}")


if __name__ == "__main__":
    import argparse
    import csv
    import json
    from math import isfinite

    parser = argparse.ArgumentParser(description="查询单股原始因子")
    parser.add_argument("symbols", nargs="*", default=["601868"], help="一个或多个股票代码或精确名称")
    parser.add_argument("--end-date", help="日线截止日，格式 YYYY-MM-DD")
    parser.add_argument("--raw-daily", action="store_true", help="额外打印计算用的整段 BaoStock 原始日线（CSV 格式）")
    args = parser.parse_args()
    # 支持一次传入多个代码，依次打印每只股票的结果。
    for symbol in args.symbols:
        result = fetch_stock_factor_vector(symbol, args.end_date, include_daily_history=args.raw_daily)
        # 20 是字段总数。单独统计空值和非有限数，让终端一眼能看出取到了几项。
        missing_factors = [
            name for name, value in result["vector"].items()
            if value is None or not isfinite(value)
        ]
        print(json.dumps({
            "success": result["success"],
            "symbol": result["symbol"],
            "data_date": result["data_date"],
            "factor_count": len(result["vector_values"]),
            "valid_factor_count": len(result["vector_values"]) - len(missing_factors),
            "missing_factors": missing_factors,
        }, ensure_ascii=False, indent=2))
        print("\n20 因子向量（英文缩写 / 数值 / 中文指标）")
        for name, value in result["vector"].items():
            # 数值保留四位小数。缺失时显示 NULL，这一行仍会打印对应的中文名称。
            display = "NULL" if name in missing_factors else f"{value:.4f}"
            print(f"  {name:<9} {display:>10}  {FACTOR_LABELS[name]}")

        _print_source_metadata(result)
        if args.raw_daily:
            print(f"\n{result['symbol']} BaoStock 原始日线（CSV）")
            history = result["daily_history"]
            writer = csv.writer(sys.stdout, lineterminator="\n")
            writer.writerow(history["fields"])
            writer.writerows(history["rows"])
