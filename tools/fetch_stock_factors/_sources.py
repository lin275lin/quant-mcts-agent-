"""获取日线、估值和财务字段，再整理成计算因子需要的格式。"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from config import LOOKBACK_CALENDAR_DAYS

from ._cache import _cache_read, _cache_write
from ._helpers import _VALUATION_COLUMNS, _finite_float
from .baostock_session import _query_baostock


_LOGGER = logging.getLogger(__name__)
# 保护日线缓存的“先查、未命中再写”过程。BaoStock 连接本身由会话模块保护。
_DAILY_FETCH_LOCK = Lock()
# 字段顺序要和接口回传的每一行数据相对应。
_DAILY_FIELDS = "date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,psTTM"
_ADJUST_FLAG = "2"  # BaoStock 前复权
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turn"]


def _request_daily_bars(symbol: str, start: str, end: str) -> dict[str, Any]:
    """准备请求参数。登录、读取结果和退出是会话模块的职责。"""
    # 将 600519.SH 改成接口认识的 sh.600519。
    code, exchange = symbol.split(".")
    # lambda 创建一个简短的函数。会话模块拿到 SDK 对象后才运行冒号右边的调用。
    payload = _query_baostock(
        lambda sdk: sdk.query_history_k_data_plus(
            f"{exchange.lower()}.{code}", _DAILY_FIELDS,
            start_date=start, end_date=end, frequency="d", adjustflag=_ADJUST_FLAG,
        )
    )
    if not payload["rows"]:
        raise RuntimeError(f"BaoStock 返回空日线: {symbol}")
    return payload


def _daily_payload(symbol: str, start: str, end: str) -> dict[str, Any]:
    """优先从缓存拿原始表格。缓存不存在时取接口并存起来。"""
    # 日期、复权方式或字段变了，就换一个键，防止取到其他条件下的旧数据。
    key = ("daily", symbol, start, end, _ADJUST_FLAG, _DAILY_FIELDS)
    payload = _cache_read(*key)
    if payload is None:
        with _DAILY_FETCH_LOCK:
            # 等锁时其他线程可能已写入相同的结果，所以这里再确认一次。
            payload = _cache_read(*key)
            if payload is None:
                payload = _request_daily_bars(symbol, start, end)
                _cache_write(payload, *key)
    return payload


def _parse_daily_bars(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """用接口列名和数据行重建日线表，并取最后一行的估值。"""
    # DataFrame 是一张可按列处理的表。接口多数数字以文本返回，需先转成数字。
    frame = pd.DataFrame(payload["rows"], columns=payload["fields"])
    for column in _PRICE_COLUMNS + _VALUATION_COLUMNS:
        # 两个列表用 + 拼起来。errors="coerce" 把转不动的文字变成 NaN，表示缺失。
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    # 先按日期排序，再用日期作为行标签。iloc[-1] 取最后一行。
    frame = frame.sort_values("date").set_index("date")
    latest = frame.iloc[-1]
    valuations = {column: _finite_float(latest[column]) for column in _VALUATION_COLUMNS}
    # 逗号分隔的两个结果按顺序返回。rename 把换手率统一命名为 turnover。
    return frame[_PRICE_COLUMNS].rename(columns={"turn": "turnover"}), valuations


def _fetch_daily_bars(symbol: str, end: date) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """按日期取日线和估值。查询及整理表格各有自己的函数。"""
    # 日期减去 timedelta 会向前推一定数量的自然日，isoformat() 转成接口所需的文字。
    start = (end - timedelta(days=LOOKBACK_CALENDAR_DAYS)).isoformat()
    payload = _daily_payload(symbol, start, end.isoformat())
    return _parse_daily_bars(payload)


def _fetch_size(code: str) -> float | None:
    """百度的总市值（亿元）转成元，再取自然对数。失败时可以返回缺失。"""
    # 同一只股票的总市值有自己的缓存键，和日线、财务数据分开保存。
    key = ("size", code)
    cached = _cache_read(*key)
    if cached is not None:
        return cached

    try:
        import akshare as ak

        # 接口返回多行时，目前用 value 列的最后一行。
        frame = ak.stock_zh_valuation_baidu(symbol=code, indicator="总市值", period="day")
        if frame is not None and not frame.empty:
            market_cap = _finite_float(frame["value"].iloc[-1])
            if market_cap is not None and market_cap > 0:
                # 百度的值以亿元表示。先乘一亿转成元，再用 np.log() 取自然对数。
                size = float(np.log(market_cap * 1e8))
                _cache_write(size, *key)
                return size
    except Exception:
        # 这个数据源失效时记录原因，下方返回 None，其他因子仍能使用。
        _LOGGER.debug("百度市值查询失败: %s", code, exc_info=True)
    return None


def _fetch_financial_factors(symbol: str) -> dict[str, float | None]:
    """按报告期读接口直接披露的加权 ROE 和营业收入同比。"""
    # 财务用带交易所的代码作键，数据缺失时仍为两个因子分别保留位置。
    key = ("financial", symbol)
    cached = _cache_read(*key)
    if cached is not None:
        return cached

    import akshare as ak

    frame = ak.stock_financial_analysis_indicator_em(symbol=symbol, indicator="按报告期")
    if frame is None or frame.empty:
        return {"ROE": None, "REVG": None}

    # ascending=False 是降序。按报告期从新到旧排列，iloc[0] 就是第一行。
    latest = frame.sort_values("REPORT_DATE", ascending=False).iloc[0]
    # .get() 找不到字段时返回 None。ROE 除以 100 转成小数，营收同比按接口值返回。
    roe = _finite_float(latest.get("ROEJQ"))
    growth = _finite_float(latest.get("TOTALOPERATEREVETZ"))
    result = {"ROE": roe / 100 if roe is not None else None, "REVG": growth}
    # 至少有一项成功取到才缓存。如果两项都没有，下次仍会查接口。
    if roe is not None or growth is not None:
        _cache_write(result, *key)
    return result
