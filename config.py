"""
全局配置：20 因子列表（按论文草稿 3.3 节表 3-1 定义）。

顺序固定——工具计算、下游加权、MCTS 搜索全部按这个顺序读。
"""
from __future__ import annotations


# 20 个因子，按分组排列。
# 每个因子的定义见 docs/papers/第x章-(草稿)因子选取与定义.txt §3.4
FACTOR_NAMES = [
    # ---- A 动量与反转（4）----
    "MOM20",      # F01  20 日动量   = C(t)/C(t-20) - 1
    "MOM60",      # F02  60 日动量
    "MOM120",     # F03  120 日动量
    "REV5",       # F04  5 日反转     = -(C(t)/C(t-5) - 1)

    # ---- B 波动（3）----
    "VOL20",      # F05  20 日年化波动率
    "VOL60",      # F06  60 日年化波动率
    "DVOL20",     # F07  20 日下行波动率

    # ---- C 流动性与规模（3）----
    "TURN20",     # F08  20 日平均换手率
    "AMIHUD20",   # F09  Amihud 非流动性
    "SIZE",       # F10  总市值的自然对数，市值先换成元

    # ---- D 量价形态（3）----
    "INTRA20",    # F11  20 日日内动量均值
    "CORR20",     # F12  20 日量价相关系数
    "VR",         # F13  量比 = 5日均量/20日均量

    # ---- E 技术指标（2）----
    "RSI14",      # F14  RSI(14) Wilder 平滑
    "AMP20",      # F15  20 日振幅均值 = mean((H-L)/O, 20)

    # ---- F 基本面（5）----
    "EP",         # F16  盈利收益率   = 1/PE_TTM
    "BP",         # F17  账面市值比   = 1/PB
    "SP",         # F18  营收市值比   = 1/PS_TTM
    "ROE",        # F19  净资产收益率（加权）
    "REVG",       # F20  营收同比增速
]


# 默认权重：等权兜底。MCTS 搜不到 / 超时 / 失败时用。
DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 / len(FACTOR_NAMES) for name in FACTOR_NAMES}


# 最新因子快照的日线抓取范围，单位是自然日。
# MOM120 加完整的 40 个历史因子值至少需要 161 根有效日线，评分历史不足时需补齐。
LOOKBACK_CALENDAR_DAYS = 275
