"""获取日线、估值和财务字段，再整理成计算因子需要的格式。"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from threading import Lock
from typing import Any

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


def _parse_daily_bars(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """还原日线表，并保留最后一行的行情和估值原值供核对。"""
    # DataFrame 是一张可按列处理的表。接口多数数字以文本返回，需先转成数字。
    frame = pd.DataFrame(payload["rows"], columns=payload["fields"])
    for column in _PRICE_COLUMNS + _VALUATION_COLUMNS:
        # 两个列表用 + 拼起来。errors="coerce" 把转不动的文字变成 NaN，表示缺失。
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    # 先按日期排序，再用日期作为行标签。iloc[-1] 取最后一行。
    frame = frame.sort_values("date").set_index("date")
    latest = frame.iloc[-1]
    latest_values: dict[str, Any] = {
        column: _finite_float(latest[column]) for column in _PRICE_COLUMNS + _VALUATION_COLUMNS
    }
    latest_values["date"] = latest.name.date().isoformat()
    # 逗号分隔的两个结果按顺序返回。rename 把换手率统一命名为 turnover。
    return frame[_PRICE_COLUMNS].rename(columns={"turn": "turnover"}), latest_values


def _fetch_daily_bars(symbol: str, end: date) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """返回计算用日线、核对信息和接口回传的整段历史。"""
    # 日期减去 timedelta 会向前推一定数量的自然日，isoformat() 转成接口所需的文字。
    start = (end - timedelta(days=LOOKBACK_CALENDAR_DAYS)).isoformat()
    end_text = end.isoformat()
    payload = _daily_payload(symbol, start, end_text)
    bars, latest = _parse_daily_bars(payload)
    metadata = {
        "source": "BaoStock",
        "adjustment": "前复权",
        "requested_start": start,
        "requested_end": end_text,
        "first_date": bars.index[0].date().isoformat(),
        "row_count": len(bars),
        "latest": latest,
    }
    return bars, metadata, payload


def _source_date(value: Any) -> str | None:
    """把来源给出的日期统一成 YYYY-MM-DD。缺失和无法识别的日期留空。"""
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else None


def _fetch_market_cap(code: str) -> dict[str, Any]:
    """读取百度股市通的总市值原值（亿元）和该条记录的日期。"""
    # 旧的 size 缓存只保留对数。这里换一个键，保存原值和日期。
    key = ("market-cap-raw", code)
    cached = _cache_read(*key)
    if cached is not None:
        return cached

    result = {"source": "百度股市通", "date": None, "value_100m_cny": None}
    try:
        import akshare as ak

        # 沿用原来取最后一行的逻辑。日期和总市值必须拿同一行。
        frame = ak.stock_zh_valuation_baidu(symbol=code, indicator="总市值", period="day")
        if frame is not None and not frame.empty:
            latest = frame.iloc[-1]
            market_cap = _finite_float(latest["value"])
            result["date"] = _source_date(latest.get("date"))
            result["value_100m_cny"] = market_cap
            if market_cap is not None and market_cap > 0:
                _cache_write(result, *key)
    except Exception:
        # 这个数据源失效时记录原因，下方返回 None，其他因子仍能使用。
        _LOGGER.debug("百度市值查询失败: %s", code, exc_info=True)
    return result


def _fetch_financial_indicators(symbol: str) -> dict[str, Any]:
    """按报告期读取东方财富披露的百分数，以及对应的报告期。"""
    # 旧的 financial-ratios 键保存的是换算后的小数。新键保留接口百分数和日期。
    key = ("financial-raw", symbol)
    cached = _cache_read(*key)
    if cached is not None:
        return cached

    import akshare as ak

    result = {
        "source": "东方财富",
        "report_date": None,
        "announcement_date": None,
        "ROEJQ": None,
        "TOTALOPERATEREVETZ": None,
    }
    frame = ak.stock_financial_analysis_indicator_em(symbol=symbol, indicator="按报告期")
    if frame is None or frame.empty:
        return result

    # ascending=False 是降序。按报告期从新到旧排列，iloc[0] 就是第一行。
    latest = frame.sort_values("REPORT_DATE", ascending=False).iloc[0]
    # .get() 找不到字段时返回 None。公告日从原表读取，有值才记录。
    roe = _finite_float(latest.get("ROEJQ"))
    growth = _finite_float(latest.get("TOTALOPERATEREVETZ"))
    result.update(
        report_date=_source_date(latest.get("REPORT_DATE")),
        announcement_date=_source_date(latest.get("NOTICE_DATE")),
        ROEJQ=roe,
        TOTALOPERATEREVETZ=growth,
    )
    # 至少有一项成功取到才缓存。如果两项都没有，下次仍会查接口。
    if roe is not None or growth is not None:
        _cache_write(result, *key)
    return result
