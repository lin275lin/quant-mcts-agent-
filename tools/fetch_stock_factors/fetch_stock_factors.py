"""单股因子工具的公开入口。

第一次读代码可以先找下面的 fetch_stock_factor_vector 函数。
它安排取数据的顺序，并把取到的结果拼起来。接口请求在 _sources.py，因子公式在 _helpers.py。
以 # 开头的是注释，三引号中的说明叫文档字符串，两者都可以用来了解代码。
"""

# 类型标注用来帮助阅读代码。这个导入让 Python 先保留标注，等需要时再解释。
from __future__ import annotations

# import 表示借用已经写好的代码。下面三个工具分别负责并行、日期和类型标注。
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

# if 开头是一条判断。缩进到 if 下面的语句，只有判断成立才运行。
# __package__ 由 Python 设置。直接运行这个文件时，它通常是空的。
if not __package__:
    import sys
    from pathlib import Path

    # __file__ 是这个文件的路径。parents[2] 往上找到 Project 目录。
    # sys.path 是 Python 查模块的目录列表，insert(0, ...) 把 Project 放到最前面。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.fetch_stock_factors import _helpers, _sources, _symbols
else:
    # 如果这是被别处导入的模块，点号 . 表示从当前包中导入同目录的模块。
    from . import _helpers, _sources, _symbols

# 因子的名字和顺序由 config.py 统一定义，其他模块可以按同一顺序读列表。
from config import FACTOR_NAMES


# def 定义一个函数，括号中是调用者传给它的参数。
# 参数的冒号后面是类型提示，= None 表示可以不传这个参数。
# str | None 表示可以是字符串或空值，-> 右边提示返回什么类型。
def fetch_stock_factor_vector(
    symbol: str,
    data_end_date: str | None = None,
) -> dict[str, Any]:
    """返回单股最近可用日的 20 个原始因子。

    接受股票代码、带交易所的代码或精确股票名称。指定截止日会限制日线，
    市值与财务仍是各自数据源的最新快照。
    """
    # “A if 条件 else B”表示条件成立选 A，否则选 B，这里二选一得到截止日期。
    end = date.fromisoformat(data_end_date) if data_end_date else date.today()
    # _resolve() 把不同写法统一成类似 600519.SH 的格式。
    standard_symbol = _symbols._resolve(symbol)
    # split(".") 在点号处分开，左边给 code，右边给 _。这里的 _ 表示后面不用它。
    code, _ = standard_symbol.split(".")

    # ThreadPoolExecutor 可以让两个工作线程同时等待网络。
    # with 管理这些线程的生命周期，离开缩进范围时会收尾。
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stock-factors") as workers:
        # submit(函数, 参数) 安排在线程里调用它，先返回一个等结果的凭据 Future。
        size_future = workers.submit(_sources._fetch_size, code)
        financial_future = workers.submit(_sources._fetch_financial_factors, standard_symbol)
        # 当前线程同时去取 BaoStock 日线。辅助函数一次返回日线表和当天估值两个结果。
        bars, valuations = _sources._fetch_daily_bars(standard_symbol, end)
        price_factors = _helpers._compute_price_factors(bars)
        # 转成 {因子名称: 数值} 的字典。update() 把另一份字典中的项目合并进来。
        factors = price_factors.to_dict()
        factors.update(_helpers._valuation_factors(valuations))
        # result() 真正拿到线程计算的结果。如果线程还没完成，这里会等它。
        factors["SIZE"] = size_future.result()
        factors.update(financial_future.result())

    # 这是字典推导式，相当于按 FACTOR_NAMES 循环，把每个名称和数值写入新字典。
    # .get(name) 找不到该名称时会返回 None，因子仍然有自己的位置。
    vector = {name: factors.get(name) for name in FACTOR_NAMES}
    # return 把结果交还给调用函数的人。下面每一行都是一个“名字: 内容”。
    return {
        "success": True,
        "symbol": standard_symbol,
        # pandas 用日期给结果行命名。这里把日期转成 YYYY-MM-DD 格式的字符串。
        "data_date": price_factors.name.date().isoformat(),
        "vector": vector,
        # values() 只拿数值。list() 把它们装成按因子顺序排列的列表。
        "vector_values": list(vector.values()),
        # copy() 返回一份名单副本，外部改这份副本时不会改到全局配置。
        "factor_names": FACTOR_NAMES.copy(),
    }


# 只有直接运行这个 .py 文件时，__name__ 才是 "__main__"，下方打印代码才会执行。
if __name__ == "__main__":
    import argparse
    import json
    from math import isfinite

    # argparse 负责读终端输入。nargs="?" 表示股票参数可以省略，default 给出默认值。
    parser = argparse.ArgumentParser(description="查询单股原始因子")
    parser.add_argument("symbol", nargs="?", default="600519", help="股票代码或精确名称")
    parser.add_argument("--end-date", help="日线截止日，格式 YYYY-MM-DD")
    # 解析后的参数可以用 args.symbol 和 args.end_date 访问。
    args = parser.parse_args()
    result = fetch_stock_factor_vector(args.symbol, args.end_date)
    # dumps() 把 Python 字典变成 JSON 文本。indent=2 让输出缩进，方便在终端阅读。
    print(json.dumps({
        "success": result["success"],
        "symbol": result["symbol"],
        "data_date": result["data_date"],
        "factor_count": len(result["vector_values"]),
    }, ensure_ascii=False, indent=2))
    # \n 是换行。items() 同时交出字典中每一项的名称和数值。
    print("\n20 因子向量（按 FACTOR_NAMES 顺序）:")
    for name, value in result["vector"].items():
        # f 字符串会把花括号里的结果填进文本。>10.4f 表示保留 4 位小数并右对齐。
        # 缺失值和无穷等非有限数显示为 NULL。
        display = f"{value:>10.4f}" if value is not None and isfinite(value) else "      NULL"
        # <14 表示因子名在至少 14 个字符的宽度里左对齐。
        print(f"  {name:<14} {display}")
