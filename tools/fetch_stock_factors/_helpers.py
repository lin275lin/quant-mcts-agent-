"""纯因子计算。这里只用传入的数据计算，不请求接口也不打开数据库。"""

from __future__ import annotations

from typing import Any

# as 为模块取简称。NumPy 做数值计算，pandas 用来整理和计算表格。
import numpy as np
import pandas as pd


_VALUATION_COLUMNS = ["peTTM", "pbMRQ", "psTTM"]


def _momentum_factors(close: pd.Series) -> dict[str, pd.Series]:
    """A 组，按交易观测计算动量和短期反转。"""
    # pct_change(20) 表示当前值除以 20 行前的值再减 1。反转因子将涨跌幅取反。
    return {
        "MOM20": close.pct_change(20),
        "MOM60": close.pct_change(60),
        "MOM120": close.pct_change(120),
        "REV5": -close.pct_change(5),
    }


def _volatility_factors(returns: pd.Series) -> dict[str, pd.Series]:
    """B 组，年化日收益波动与下行波动。"""
    # 项目按每年 252 个交易日换算。std() 是样本标准差，min_periods 是最少有效值个数。
    annualization = np.sqrt(252)
    # where(条件, 替代值) 把不满足条件的项改为替代值，这里只保留负收益。
    down = returns.where(returns < 0, 0.0)
    return {
        "VOL20": returns.rolling(20, min_periods=10).std() * annualization,
        "VOL60": returns.rolling(60, min_periods=30).std() * annualization,
        "DVOL20": down.rolling(20, min_periods=10).std() * annualization,
    }


def _liquidity_factors(bars: pd.DataFrame, returns: pd.Series) -> dict[str, pd.Series]:
    """C 组，换手率和 Amihud 非流动性。"""
    # 成交额为零时记为缺失，避免除以零。1e8 等于一亿，用来缩放很小的比值。
    amount = bars["amount"].replace(0, np.nan)
    return {
        "TURN20": bars["turnover"].rolling(20, min_periods=10).mean(),
        "AMIHUD20": (returns.abs() / amount).rolling(20, min_periods=10).mean() * 1e8,
    }


def _volume_price_factors(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """D 组，日内涨跌、量价相关与量比。"""
    close, opening, volume = bars["close"], bars["open"], bars["volume"]
    # rolling(20) 对截至当日最近 20 行计算一次。corr() 计算价格和成交量的相关系数。
    return {
        "INTRA20": ((close - opening) / opening).rolling(20, min_periods=10).mean(),
        "CORR20": close.rolling(20).corr(volume),
        "VR": volume.rolling(5).mean() / volume.rolling(20).mean(),
    }


def _relative_strength_index(close: pd.Series) -> pd.Series:
    """将平滑后的平均上涨和下跌幅度换算为 RSI。"""
    # diff() 求当天与前一天的价差。把上涨和下跌分别放入两列。
    change = close.diff()
    gain = change.where(change > 0, 0.0)
    loss = -change.where(change < 0, 0.0)
    # ewm() 做递推平均。较新的数据影响更大，这里沿用 Wilder 的 1/14 权重。
    average_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    # 平均跌幅为零时，当前公式按缺失处理。
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative_strength)


def _technical_factors(bars: pd.DataFrame) -> dict[str, pd.Series]:
    """E 组，RSI 和平均振幅。"""
    # 每天用高低价之差除以开盘价，再算 20 日平均。
    amplitude = (bars["high"] - bars["low"]) / bars["open"]
    return {
        "RSI14": _relative_strength_index(bars["close"]),
        "AMP20": amplitude.rolling(20, min_periods=10).mean(),
    }


def _compute_price_factors(bars: pd.DataFrame) -> pd.Series:
    """按组算完 14 个价量因子，再取最近一个有数据的交易日。"""
    # 表名["列名"] 提取一整列，pandas 把它叫 Series。pct_change() 默认和上一行相比。
    close = bars["close"]
    returns = close.pct_change()
    # ** 把各组字典展开到同一个字典。用日期索引把各列对齐。
    factors = pd.DataFrame({
        **_momentum_factors(close),
        **_volatility_factors(returns),
        **_liquidity_factors(bars, returns),
        **_volume_price_factors(bars),
        **_technical_factors(bars),
    }, index=bars.index)
    # 去掉所有因子都缺失的行。iloc[-1] 取表里的最后一行。
    valid = factors.dropna(how="all")
    return valid.iloc[-1] if not valid.empty else factors.iloc[-1]


def _finite_float(value: Any) -> float | None:
    """把能用的输入转成浮点数，缺失或无效的输入统一返回 None。"""
    # pd.isna() 识别 None、NaN 之类的缺失值。
    if pd.isna(value):
        return None
    try:
        # float() 能把 "12.5" 这样的字符串转成数字，文字无法转换时会报错。
        number = float(value)
    except (TypeError, ValueError):
        return None
    # isfinite() 检查是不是有限数，将无穷和 NaN 排除在外。
    return number if np.isfinite(number) else None


def _valuation_factors(valuations: dict[str, float | None]) -> dict[str, float | None]:
    """EP、BP、SP 随日线返回，仅对有效的正估值倍数取倒数。"""
    factors = {}
    # zip() 按位置把两列名字配对，比如 EP 对 peTTM。for 每次拿出一对。
    for name, column in zip(("EP", "BP", "SP"), _VALUATION_COLUMNS):
        value = valuations[column]
        # 先确认有数，再判断大于 0。and 左边不通过时，不会继续比较右边。
        factors[name] = 1.0 / value if value is not None and value > 0 else None
    return factors
