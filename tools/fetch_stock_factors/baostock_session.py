"""BaoStock 的登录、退出和连接管理。这里不计算股票因子。"""

from __future__ import annotations

import atexit
import logging
from socket import socket
from threading import Lock
from types import ModuleType
from typing import Any, Callable


_LOGGER = logging.getLogger(__name__)
# BaoStock 的 SDK 在一个进程里共用一个连接。锁让登录、查询和退出依次执行。
_SESSION_LOCK = Lock()
_SOCKET_TIMEOUT_SECONDS = 30
_NETWORK_ERRORS = {"10001001"} | {f"1000200{number}" for number in range(1, 9)}
_keep_open = False


class _DisconnectedError(RuntimeError):
    """专门标记连接故障，让程序能区分是否应该重连。"""


class _SocketGuard:
    """包装 SDK 的连接，避免服务器已断开时一直读到空数据。"""

    def __init__(self, connection: socket) -> None:
        # self 代表这个对象。加上 self. 后，这些值可在它的其他方法里访问。
        self.connection = connection
        self.failed = False
        # 登录成功之后才设置读写超时。这个配置只影响这一条连接。
        self.connection.settimeout(_SOCKET_TIMEOUT_SECONDS)

    def send(self, data: bytes) -> int:
        try:
            self.connection.sendall(data)
        except OSError:
            self.failed = True
            raise  # 没写新错误时，raise 会继续传递刚刚捕获的原错误。
        return len(data)

    def recv(self, size: int) -> bytes:
        try:
            data = self.connection.recv(size)
            # b"" 是空字节，表示对端已经关闭连接。
            if not data:
                raise ConnectionResetError("BaoStock 已关闭连接")
            return data
        except OSError:
            self.failed = True
            raise

    def close(self) -> None:
        self.connection.close()


_session: _SocketGuard | None = None


def _open_session() -> ModuleType:
    """调用者先拿锁。连接已存在时，直接返回 SDK。"""
    import baostock as bs
    from baostock.common import context

    global _session
    if _session is None:
        login = bs.login()
        raw_socket = getattr(context, "default_socket", None)
        if login.error_code != "0":
            # 某些失败会留下一条没登录的连接，先将它关闭。
            if raw_socket is not None:
                raw_socket.close()
            context.default_socket = None
            raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
        _session = _SocketGuard(raw_socket)
        context.default_socket = _session
    return bs


def _close_session(*, graceful: bool) -> None:
    """调用者先拿锁。只有健康连接才发送正常登出请求。"""
    from baostock.common import context

    global _session
    connection = _session
    if connection is None:
        return
    try:
        if graceful and not connection.failed:
            import baostock as bs

            bs.logout()
    finally:
        # 即使登出报错，也清空状态并确保本地套接字已关闭。
        _session = None
        context.default_socket = None
        connection.close()


def login_baostock() -> None:
    """提前登录并保留连接，供这个进程后续的多次查询复用。"""
    global _keep_open
    with _SESSION_LOCK:
        _open_session()
        _keep_open = True


def logout_baostock() -> None:
    """结束主动保留的连接。没有连接时，可安全地再次调用。"""
    global _keep_open
    with _SESSION_LOCK:
        _keep_open = False
        if _session is not None:
            _close_session(graceful=True)


@atexit.register
def _logout_at_exit() -> None:
    # @ 表示装饰器。这里将函数登记给 Python，进程正常退出时自动调用。
    try:
        logout_baostock()
    except OSError:
        _LOGGER.debug("退出时关闭 BaoStock 连接失败", exc_info=True)


def _read_rows(query: Callable[[ModuleType], Any]) -> dict[str, Any]:
    """执行一次查询，并把结果对象转换成能写到缓存的列名和行数据。"""
    result = query(_open_session())
    rows = []
    # while 先检查状态成功，再问 SDK 有没有下一行，两项都成立才读取。
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    # SDK 在无法拿到后续页时可能只返回 False，因此也看套接字是否报过错。
    if _session.failed or result.error_code in _NETWORK_ERRORS:
        raise _DisconnectedError(f"BaoStock 查询连接失败: {result.error_msg}")
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock 查询失败: {result.error_msg}")
    return {"fields": result.fields, "rows": rows}


def _query_baostock(query: Callable[[ModuleType], Any]) -> dict[str, Any]:
    """调用方提供一个查询函数，连接由本模块统一管理。"""
    with _SESSION_LOCK:
        attempts = 2 if _keep_open else 1
        try:
            for attempt in range(attempts):
                try:
                    return _read_rows(query)
                except _DisconnectedError:
                    # 坏连接直接关闭。还有机会时，下一次循环会重新登录。
                    _close_session(graceful=False)
                    if attempt == attempts - 1:
                        raise
        finally:
            # 日常命令行查询仍然用完就退出。常驻进程通过 login_baostock 保留连接。
            if not _keep_open and _session is not None:
                _close_session(graceful=True)
