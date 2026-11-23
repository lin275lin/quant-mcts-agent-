"""股票数据的 SQLite 磁盘缓存。其他模块通过两个函数读写。"""

from __future__ import annotations

# json 转换 Python 数据和文本，sqlite3 读写磁盘数据库，time 用来判断是否过期。
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any


# __name__ 是模块名。日志带上它，出现问题时就能看出消息来自哪个文件。
_LOGGER = logging.getLogger(__name__)
# Path 的 / 用来拼目录，和数字相除的用法不同。
# parents[2] 从 _helpers.py 向上找到 Project，因此从别的目录运行，缓存位置也相同。
_CACHE_PATH = Path(__file__).resolve().parents[2] / "storage" / "cache" / "stock_factor_sources.sqlite3"
# 保存 15 分钟，程序内部统一用秒。版本号也会写进缓存键，用来区分不同口径的数据。
_CACHE_TTL_SECONDS = 15 * 60
_CACHE_VERSION = "v1"


def _connect_cache() -> sqlite3.Connection:
    """打开缓存数据库，返回一次读写所用的连接。"""
    # parent 是文件所在目录。parents=True 允许把缺少的上级目录一起创建。
    # exist_ok=True 让目录已存在时也能继续执行。
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 数据库文件不存在时，connect() 会创建它。timeout 是等待磁盘库解锁的秒数。
    return sqlite3.connect(_CACHE_PATH, timeout=0.25)


def _cache_read(*parts: str) -> Any:
    """读取未过期的源数据。缓存不可用时退回正常查询。"""
    # 参数名前有 *，表示可以传任意多个位置参数，函数会把它们收成一个元组。
    # 元组是一组有顺序的值。这里再用 *parts 把这些值展开，与版本号一起编码为缓存键。
    # ensure_ascii=False 让键里的中文能按原样保存，方便有人打开数据库时阅读。
    key = json.dumps((_CACHE_VERSION, *parts), ensure_ascii=False)
    # try 表示先试着执行下面的代码。碰到指定错误时，再走 except 中的处理。
    try:
        # 这里的 with 会在这一段结束时关闭数据库连接。
        with closing(_connect_cache()) as connection:
            # execute() 发送 SQL 指令。建表语句只在表还不存在时起作用。
            # key 是唯一查找编号，expires_at 是过期时间，payload 是实际的缓存内容。
            connection.execute(
                "CREATE TABLE IF NOT EXISTS source_cache "
                "(key TEXT PRIMARY KEY, expires_at REAL NOT NULL, payload TEXT NOT NULL)"
            )
            # SELECT 查数据。? 是占位符，数据库会用后面的查询键和当前时间替换它们。
            # 仅选出未过期的一行，fetchone() 找不到时会返回 None。
            row = connection.execute(
                "SELECT payload FROM source_cache WHERE key = ? AND expires_at > ?",
                (key, time.time()),
            ).fetchone()
            # row[0] 是查到的第一列。loads() 把落在硬盘上的 JSON 文本还原成 Python 数据。
            return json.loads(row[0]) if row is not None else None
    except (OSError, sqlite3.Error, ValueError):
        # 记录文件、SQLite 或 JSON 读取错误，返回 None，让上游从接口取数据。
        _LOGGER.debug("读取因子数据缓存失败", exc_info=True)
        return None


def _cache_write(payload: Any, *parts: str, ttl: int = _CACHE_TTL_SECONDS) -> None:
    """把数据写入缓存。ttl 表示还可以使用多少秒。"""
    # ttl 写在 *parts 后面，调用时要用 ttl=... 指定。省略时用默认的 15 分钟。
    key = json.dumps((_CACHE_VERSION, *parts), ensure_ascii=False)
    try:
        # 第一个上下文负责关闭连接，第二个负责这次写入。成功就提交，遇到异常会撤销。
        with closing(_connect_cache()) as connection, connection:
            # time.time() 用秒表示当前时间，所以加上 ttl 就得到过期时间。
            now = time.time()
            # 先清掉过期行，再写入新行。相同 key 已存在时，SQL 会用新内容替换原行。
            connection.execute("DELETE FROM source_cache WHERE expires_at <= ?", (now,))
            # VALUES 中的三个 ? 分别对应键、过期时间和 JSON 内容。allow_nan=False 拒绝非标准 JSON 数值。
            connection.execute(
                "INSERT OR REPLACE INTO source_cache VALUES (?, ?, ?)",
                (key, now + ttl, json.dumps(payload, ensure_ascii=False, allow_nan=False)),
            )
    except (OSError, sqlite3.Error, ValueError):
        # 缓存只是加速。写入失败时，调用者仍然可以使用刚从接口取得的结果。
        _LOGGER.debug("写入因子数据缓存失败", exc_info=True)
