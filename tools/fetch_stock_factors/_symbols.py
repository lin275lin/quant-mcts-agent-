"""把输入的股票名称或不同格式的代码变成统一的代码。"""

from __future__ import annotations

import re
from threading import Lock

from ._cache import _cache_read, _cache_write


# re.compile() 把匹配规则准备好。\d{6} 表示六位数字，[A-Z]{2} 表示两个字母。
# 第一条匹配 600519 或 600519.SH，第二条匹配 sh600519。IGNORECASE 表示不区分大小写。
_SYMBOL_RE = re.compile(r"(\d{6})(?:\.([A-Z]{2}))?", re.IGNORECASE)
_PREFIX_RE = re.compile(r"(SH|SZ|BJ)(\d{6})", re.IGNORECASE)
# dict[str, str] 表示字典的键和值都是字符串。None 表示名称表还没加载。
_NAME_MAP: dict[str, str] | None = None
_NAME_LOCK = Lock()


def _infer_suffix(code: str) -> str:
    """用六位代码的第一个字符推断交易所后缀。"""
    # 索引从 0 开始，所以 code[0] 是首字符。“in”判断它是否在指定字符串中。
    if code[0] in "03":
        return "SZ"
    if code[0] in "489":
        return "BJ"
    return "SH"


def _resolve(raw: str) -> str:
    """代码在本地解析。只有精确匹配股票名称时才查询名称表。"""
    # raw 为空时先用空字符串代替。strip() 去掉首尾空格。
    raw = (raw or "").strip()
    if not raw:
        # raise 主动抛出错误，函数在这里结束。调用者可以用 try/except 处理。
        raise ValueError("股票代码或名称不能为空")
    # := 先把匹配结果存进 match 再判断。fullmatch() 要求输入的整段文字符合格式。
    if match := _SYMBOL_RE.fullmatch(raw):
        # groups() 按正则括号的顺序取值。两个值分别存入 code 和 suffix。
        code, suffix = match.groups()
        # f 字符串用 {} 填值。已有后缀就转成大写，缺后缀就根据代码推断。
        return f"{code}.{suffix.upper() if suffix else _infer_suffix(code)}"
    if match := _PREFIX_RE.fullmatch(raw):
        suffix, code = match.groups()
        return f"{code}.{suffix.upper()}"

    # 两种代码写法都匹配不上时，再按名称查。global 表示更新文件顶部那份共享字典。
    global _NAME_MAP
    # 进入 with 时拿锁，出去时释放。两个线程同时查名称时不会一起去下载整张表。
    with _NAME_LOCK:
        if _NAME_MAP is None:
            # 先查磁盘上的名称表。第一次查询或缓存过期时，返回 None。
            _NAME_MAP = _cache_read("names")
            if _NAME_MAP is None:
                try:
                    import akshare as ak

                    # 表里的每一行都有 code 和 name。index=False 表示省略 pandas 的行号。
                    rows = ak.stock_info_a_code_name().itertuples(index=False)
                    names = {}
                    # for 循环一行一行取，and 要求长度和纯数字两个条件都成立。
                    for row in rows:
                        code = str(row.code).strip()
                        if len(code) == 6 and code.isdigit():
                            names[str(row.name).strip()] = f"{code}.{_infer_suffix(code)}"
                    # 空字典在 if 中算假值。只把有数据的表写进缓存。
                    if names:
                        _NAME_MAP = names
                        _cache_write(names, "names", ttl=24 * 60 * 60)
                except Exception as error:
                    # from error 保留原始报错，方便以后追查请求为何失败。
                    # 失败不写空名称表，下次查询允许重试。
                    raise ValueError(f"无法解析股票: {raw}") from error
        # in 字典判断名字是否存在，字典[名字] 取出该名字对应的股票代码。
        if _NAME_MAP is not None and raw in _NAME_MAP:
            return _NAME_MAP[raw]
    raise ValueError(f"无法解析股票: {raw}")
