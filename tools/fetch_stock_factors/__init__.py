"""把这个目录作为可导入的 Python 包，并提供简短的公开导入路径。"""

# 点号表示当前包。这一行把实现文件里的函数带到包的入口，方便外部直接导入。
from .fetch_stock_factors import fetch_stock_factor_vector
from .baostock_session import login_baostock, logout_baostock

# __all__ 写出这个包对外公开的名字，例如别人用 import * 时会得到这项。
__all__ = ["fetch_stock_factor_vector", "login_baostock", "logout_baostock"]
