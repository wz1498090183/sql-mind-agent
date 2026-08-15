"""
训练侧独立的 SQLite 只读执行工具。
与 backend/app/db_utils.py 解耦，GRPO 训练镜像不依赖推理服务代码。
"""

import os
import sqlite3
import threading
from pathlib import Path


def _get_connection(db_id: str) -> sqlite3.Connection:
    """根据 db_id 建立 SQLite 连接（复用 SPIDER_DB_ROOT 约定）。"""
    root = os.environ.get("SPIDER_DB_ROOT", "./spider_data/database")
    db_path = Path(root) / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def execute_sql(db_id: str, sql: str, timeout: float = 5.0) -> dict:
    """只读执行 SQL，返回 {success, columns, rows, error}。

    仅允许 SELECT / WITH，禁止写操作；带看门狗硬超时。
    """
    stripped = sql.strip()
    upper = stripped[:16].upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"success": False, "columns": [], "rows": [], "error": "仅允许只读查询"}

    try:
        conn = _get_connection(db_id)
    except FileNotFoundError as e:
        return {"success": False, "columns": [], "rows": [], "error": str(e)}

    interrupted = threading.Event()

    def _watchdog():
        if not interrupted.wait(timeout):
            try:
                conn.interrupt()
            except Exception:
                pass

    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        with conn:
            conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
            cursor = conn.execute(stripped)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [tuple(r) for r in cursor.fetchall()]
            return {"success": True, "columns": columns, "rows": rows, "error": None}
    except sqlite3.Error as e:
        return {"success": False, "columns": [], "rows": [], "error": str(e)}
    finally:
        interrupted.set()
        conn.close()
