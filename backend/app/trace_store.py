"""
链路追踪落库模块。
在每次请求结束后将 MainState 关键字段写入 SQLite traces 表，
便于事后 badcase 排查，不引入额外中间件。

用法:
    from app.trace_store import init_trace_db, save_trace
    init_trace_db()   # 首次启动时建表
    save_trace(state) # 请求结束时落库
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 数据库文件路径（可通过 TRACE_DB_PATH 环境变量覆盖，默认项目根目录）
_DB_PATH: Path = Path(
    os.environ.get(
        "TRACE_DB_PATH",
        str(Path(__file__).resolve().parent.parent / "traces.db"),
    )
)

# trace_id/question/db_id 列上的索引名
_IDX_TRACE_ID: str = "idx_traces_trace_id"
_IDX_QUESTION: str = "idx_traces_question"
_IDX_DB_ID: str = "idx_traces_db_id"


def init_trace_db(db_path: str | None = None) -> str:
    """初始化 traces 表，若表不存在则创建。

    建表 DDL:
        CREATE TABLE IF NOT EXISTS traces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id    TEXT NOT NULL,
            question    TEXT NOT NULL,
            db_id       TEXT NOT NULL,
            final_answer TEXT,
            status      TEXT NOT NULL DEFAULT 'unknown',
            iterations  INTEGER NOT NULL DEFAULT 0,
            plan_json   TEXT,
            error_info  TEXT,
            created_at  TEXT NOT NULL
        );

    同时在 trace_id / question / db_id 上建索引，加速排查。

    Args:
        db_path: 可选，自定义数据库文件路径，默认项目根目录 traces.db。

    Returns:
        str: 实际使用的数据库文件绝对路径。
    """
    target = Path(db_path) if db_path else _DB_PATH
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id    TEXT    NOT NULL,
                question    TEXT    NOT NULL,
                db_id       TEXT    NOT NULL,
                final_answer TEXT,
                status      TEXT    NOT NULL DEFAULT 'unknown',
                iterations  INTEGER NOT NULL DEFAULT 0,
                plan_json   TEXT,
                error_info  TEXT,
                created_at  TEXT    NOT NULL
            )
        """)
        # 创建索引（IF NOT EXISTS 在 SQLite 3.26+ 中可用）
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {_IDX_TRACE_ID}
            ON traces(trace_id)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {_IDX_QUESTION}
            ON traces(question)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {_IDX_DB_ID}
            ON traces(db_id)
        """)
        conn.commit()
    finally:
        conn.close()

    return str(target)


def _serialize_plan(plan: list[dict]) -> str | None:
    """将 plan 列表序列化为 JSON 字符串。

    若 plan 为空则返回 None。

    Args:
        plan: SubTask 列表（TypedDict 或 dict）。

    Returns:
        str | None: JSON 字符串，plan 为空时返回 None。
    """
    if not plan:
        return None
    try:
        return json.dumps(plan, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        # 兜底：逐项转字符串后序列化
        simplified = []
        for item in plan:
            if isinstance(item, dict):
                simplified.append({k: str(v) for k, v in item.items()})
            else:
                simplified.append(str(item))
        return json.dumps(simplified, ensure_ascii=False)


def save_trace(state: dict[str, Any], db_path: str | None = None) -> None:
    """将请求结束后的 MainState 关键字段写入 traces 表。

    写入字段:
        - trace_id, question, db_id: 请求标识
        - final_answer: 最终答案（可能为 None）
        - status: 执行状态（done / degraded / error）
        - iterations: 实际迭代轮次（iteration + 1）
        - plan_json: 执行计划 JSON
        - error_info: 反射失败原因（若存在）
        - created_at: 落库时间（UTC ISO 8601）

    Args:
        state: 请求完成后的 MainState 字典。
        db_path: 可选，自定义数据库文件路径。

    Raises:
        sqlite3.Error: 写入失败时抛出（调用方应捕获）。
    """
    target = Path(db_path) if db_path else _DB_PATH

    trace_id: str = state.get("trace_id", "-")
    question: str = state.get("question", "")
    db_id: str = state.get("db_id", "")
    final_answer: str | None = state.get("final_answer")
    status: str = state.get("status", "unknown")
    iteration: int = state.get("iteration", 0)
    plan: list[dict] = state.get("plan", [])

    # 提取错误信息（若存在）
    error_info: str | None = None
    reflection = state.get("reflection")
    if isinstance(reflection, dict) and not reflection.get("passed", True):
        error_info = reflection.get("reason", "")

    plan_json = _serialize_plan(plan)
    created_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(target))
    try:
        conn.execute(
            """
            INSERT INTO traces
                (trace_id, question, db_id, final_answer, status,
                 iterations, plan_json, error_info, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id, question, db_id, final_answer, status,
                iteration + 1, plan_json, error_info, created_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
