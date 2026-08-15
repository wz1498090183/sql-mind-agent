"""最小可用的日志模块，基于 loguru，仅用于调试。控制台彩色 + 文件按天切割保留7天。"""
import sys, time, uuid
from functools import wraps
from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[trace_id]}</cyan> | <level>{message}</level>")
logger.add("logs/app.log", rotation="1 day", retention="7 days", encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[trace_id]} | {message}")
logger.configure(extra={"trace_id": "-"})

def new_trace_id() -> str:
    """返回 8 位短 UUID。"""
    return uuid.uuid4().hex[:8]

def get_logger(trace_id: str = "-"):
    """返回绑定了 trace_id 的 logger。"""
    return logger.bind(trace_id=trace_id)

def log_node(func):
    """装饰器：包裹节点函数，进入/离开日志+耗时，异常打印 ERROR 并重新抛出。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = getattr(func, "__name__", "unknown")
        state = args[0] if args else {}
        log = logger.bind(trace_id=state.get("trace_id", "-") if isinstance(state, dict) else "-")
        t0 = time.perf_counter()
        log.info(f"▶ 进入节点 {name}")
        try:
            result = func(*args, **kwargs)
            log.info(f"◀ 离开节点 {name}  耗时 {(time.perf_counter() - t0) * 1000:.1f}ms")
            return result
        except Exception:
            log.error(f"✖ 节点 {name} 异常  耗时 {(time.perf_counter() - t0) * 1000:.1f}ms")
            raise
    return wrapper

# ---------- 自测 ----------
if __name__ == "__main__":
    tid = new_trace_id(); lg = get_logger(tid)
    lg.info("日志模块初始化完成"); lg.debug("调试信息"); lg.warning("警告"); lg.error("错误（模拟）")
    print(f"trace_id={tid}  日志文件见 logs/app.log")
