"""
SQL 求解子图模块。
用 LangGraph 构建可复用的 SQL 求解子图：
    generate_sql → validate_sql → execute_sql → retry_decision → (END / 重试)

每个子任务在此子图内完成 SQL 生成、校验、执行、重试闭环。
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Literal

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import sqlparse
from langgraph.graph import END, StateGraph

from app.db_utils import execute_sql, get_schema, get_schema_dict
from app.llm_client import call_text
from app.log_utils import get_logger
from app.state import SubGraphState, SubTask

# ============================================================
# SQL 清洗工具
# ============================================================
def _clean_sql(text: str) -> str:
    """从 LLM 返回的文本中提取 SQL，去掉 Markdown 代码块包裹和首尾空白。

    Args:
        text: LLM 返回的原始文本。

    Returns:
        str: 清洗后的纯 SQL 语句。
    """
    cleaned = text.strip()
    # 去掉 ```sql / ``` 代码块包裹
    match = re.search(r"```(?:sql)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


# ============================================================
# 高危 SQL 关键字拦截 — 禁止写操作
# ============================================================
_FORBIDDEN_KEYWORDS = {"DROP", "DELETE", "UPDATE", "INSERT"}

def _contains_dangerous_sql(sql: str) -> str | None:
    """检查 SQL 是否包含高危写操作关键字。

    对 tokenized 后的关键字做集合匹配，命中则返回拦截原因，
    否则返回 None。

    Args:
        sql: 待检查的 SQL 语句。

    Returns:
        str | None: 命中高危关键字时返回错误描述，否则 None。
    """
    upper_sql = sql.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        # 用词边界确保匹配到的是完整关键字而非表名列名的子串
        if re.search(rf"\b{keyword}\b", upper_sql):
            return f"SQL 包含禁止操作关键字: {keyword}，仅允许只读查询（SELECT / WITH）"
    return None


# ============================================================
# Prompt 模板
# ============================================================
_SQL_GEN_PROMPT = """你是一个 SQLite SQL 专家。根据以下信息生成一条正确的 SQL 查询语句。

## 用户需求
{description}

## 数据库 Schema
{schema}

## 上游子查询结果
{upstream_results}

## 要求
- 只输出一条 SQL 语句，以分号结尾
- 不要包含任何解释、注释或 Markdown 代码块
- 确保表名和列名与 Schema 中定义的一致
- 只允许 SELECT / WITH 查询，禁止任何写操作"""

_SQL_RETRY_PROMPT = """## 上次生成的 SQL（存在错误）
{sql}

## 错误原因
{error}

请修正以上 SQL 语句，重新生成一条正确的查询。确保只输出 SQL，不要包含任何额外文字。"""


# ============================================================
# 节点 1: generate_sql_node — 生成 SQL
# ============================================================
def _generate_sql_node(state: SubGraphState) -> dict:
    """LLM 生成 SQL 节点。

    根据 subtask 的 description、数据库 schema 和上游依赖结果拼装 Prompt，
    调用 llm_client.call_text 生成 SQL。若本次为 retry_count > 0 且存在
    上次 SQL 与错误信息，则追加修正提示。

    Args:
        state: 子图状态 SubGraphState。

    Returns:
        dict: 更新后的状态（仅含变更字段）。
    """
    subtask: SubTask = state["subtask"]
    schema: str = state.get("schema", "")
    upstream_results: dict = state.get("upstream_results", {})
    trace_id: str = state["subtask"].get("id", "-")
    log = get_logger(trace_id)

    retry_count = subtask.get("retry_count", 0)
    previous_sql = subtask.get("sql")
    previous_error = subtask.get("error")

    # 格式化上游结果
    upstream_text = ""
    if upstream_results:
        upstream_text = "\n".join(
            f"### {k}\n{v}" for k, v in upstream_results.items()
        )
    else:
        upstream_text = "无（本子任务为独立查询，无上游依赖）"

    # 基础 prompt
    prompt = _SQL_GEN_PROMPT.format(
        description=subtask["description"],
        schema=schema,
        upstream_results=upstream_text,
    )

    # 若是重试，追加上次 SQL 和错误信息
    is_retry = retry_count > 0 and previous_sql and previous_error
    if is_retry:
        prompt += "\n\n" + _SQL_RETRY_PROMPT.format(
            sql=previous_sql, error=previous_error,
        )
        log.info(
            f"generate_sql 重试  第 {retry_count} 次  "
            f"上次SQL=[{previous_sql[:100]}{'...' if len(previous_sql) > 100 else ''}]  "
            f"错误={previous_error}"
        )

    log.info(f"generate_sql 进入  需求长度={len(subtask['description'])}  schema长度={len(schema)}")

    t0 = time.perf_counter()
    raw_sql = call_text(prompt=prompt, trace_id=trace_id)
    cleaned_sql = _clean_sql(raw_sql)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    subtask["sql"] = cleaned_sql
    subtask["error"] = None  # 清除上次的 error，新 SQL 等待校验
    subtask["status"] = "running"

    log.info(
        f"generate_sql 完成  耗时 {elapsed_ms:.0f}ms  "
        f"SQL长度={len(cleaned_sql)}  "
        f"SQL=[{cleaned_sql[:120]}{'...' if len(cleaned_sql) > 120 else ''}]"
    )
    return {"subtask": subtask}


# ============================================================
# 节点 2: validate_sql_node — 校验 SQL
# ============================================================
def _validate_sql_node(state: SubGraphState) -> dict:
    """SQL 校验节点。

    执行三层校验：
        1. sqlparse 基础语法解析 — 语法不合法直接标记 error。
        2. 高危关键字拦截 — DROP/DELETE/UPDATE/INSERT 命中直接判失败。
        3. 表名存在性校验 — 通过 get_schema_dict 检查 SQL 中引用的表名。

    校验通过则 error 保持 None；失败则在 subtask.error 中写入原因。

    Args:
        state: 子图状态 SubGraphState。

    Returns:
        dict: 更新后的状态。
    """
    subtask: SubTask = state["subtask"]
    db_id: str = state.get("db_id", "")
    trace_id: str = subtask.get("id", "-")
    log = get_logger(trace_id)
    sql = subtask.get("sql", "")

    log.info(f"validate_sql 进入  SQL=[{sql[:100]}{'...' if len(sql) > 100 else ''}]")

    errors: list[str] = []

    # ---- 第 1 层: sqlparse 语法解析 ----
    try:
        parsed = sqlparse.parse(sql)
        # 过滤空语句
        statements = [s for s in parsed if s.tokens and str(s).strip()]
        if not statements:
            errors.append("SQL 解析结果为空，未检测到有效语句")
    except Exception as e:
        errors.append(f"sqlparse 语法解析失败: {e}")

    if not errors:
        stmt_str = str(statements[0]).strip()
        log.info(f"sqlparse 解析成功  type={statements[0].get_type()}  stmt={stmt_str[:80]}")

        # ---- 第 2 层: 高危关键字拦截 ----
        danger_reason = _contains_dangerous_sql(sql)
        if danger_reason:
            errors.append(danger_reason)

        # ---- 第 3 层: 表名存在性校验 ----
        try:
            schema_dict = get_schema_dict(db_id)
            existing_tables = set(schema_dict.keys())
            # 大小写不敏感地检查 SQL 中是否包含已知表名
            upper_sql = sql.upper()
            referenced = {t for t in existing_tables if t.upper() in upper_sql}
            if not referenced:
                errors.append(
                    f"SQL 中未检测到已知表名（现有表: {sorted(existing_tables)}，"
                    f"请确认 FROM / JOIN 子句中的表名是否正确）"
                )
            else:
                not_found = existing_tables - referenced
                if not_found:
                    log.info(f"表名校验通过  引用表={sorted(referenced)}（库中其他表: {sorted(not_found)}未在 SQL 中引用）")
        except Exception as e:
            log.warning(f"表名校验跳过: get_schema_dict({db_id}) 失败 — {e}")

    if errors:
        subtask["error"] = " | ".join(errors)
        log.error(f"validate_sql 失败  {subtask['error']}")
    else:
        subtask["error"] = None
        log.info("validate_sql 通过")

    return {"subtask": subtask}


# ============================================================
# 条件边: 校验后路由 — 有错误则跳 retry，无错误则进 execute
# ============================================================
def _after_validate(state: SubGraphState) -> Literal["execute_sql", "retry_decision"]:
    """校验节点后的条件边：无错误 → 执行，有错误 → 重试判断。"""
    if state["subtask"].get("error"):
        return "retry_decision"
    return "execute_sql"


# ============================================================
# 节点 3: execute_sql_node — 执行 SQL
# ============================================================
def _execute_sql_node(state: SubGraphState) -> dict:
    """SQL 执行节点。

    调用 db_utils.execute_sql 执行已校验通过的 SQL。
    成功时写入 subtask.result 并将 status 置为 "success"，
    同时写入 _completed_tasks 供主图聚合。
    失败时写入 error 并保留当前 status 供 retry_decision 判断。

    Args:
        state: 子图状态 SubGraphState。

    Returns:
        dict: 更新后的状态。
    """
    subtask: SubTask = state["subtask"]
    db_id: str = state.get("db_id", "")
    trace_id: str = subtask.get("id", "-")
    log = get_logger(trace_id)
    sql = subtask.get("sql", "")

    log.info(f"execute_sql 进入  SQL=[{sql[:100]}{'...' if len(sql) > 100 else ''}]")

    t0 = time.perf_counter()
    result = execute_sql(db_id=db_id, sql=sql)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    subtask["result"] = result

    if result["success"]:
        subtask["status"] = "success"
        subtask["error"] = None
        row_count = len(result.get("rows", []))
        log.info(
            f"execute_sql 成功  耗时 {elapsed_ms:.0f}ms  "
            f"列数={len(result.get('columns', []))}  行数={row_count}"
        )
        # 成功时加入 _completed_tasks，供主图聚合
        return {
            "subtask": subtask,
            "_completed_tasks": [subtask.copy()],
        }
    else:
        subtask["error"] = result.get("error", "未知执行错误")
        log.error(
            f"execute_sql 失败  耗时 {elapsed_ms:.0f}ms  "
            f"error={subtask['error']}"
        )
        return {"subtask": subtask}


# ============================================================
# 节点 4: retry_decision_node — 重试判断
# ============================================================
def _retry_decision_node(state: SubGraphState) -> dict:
    """重试判断节点：根据当前 status 和 retry_count 决定状态转换。

    规则：
        - status == "success" → 不动，后续路由到 END。
        - retry_count < 1 → retry_count+1，status="running"，后续回 generate_sql。
        - 否则 → status="failed"，后续路由到 END。

    Args:
        state: 子图状态 SubGraphState。

    Returns:
        dict: 更新后的状态。
    """
    subtask: SubTask = state["subtask"]
    trace_id: str = subtask.get("id", "-")
    log = get_logger(trace_id)

    # 从环境变量读取最大重试次数（默认 1 次，即共 2 次尝试）
    _sql_retry_max: int = int(os.environ.get("SQL_RETRY_MAX", "1"))

    if subtask.get("status") == "success":
        log.info("retry_decision: status=success → 路由到 END")
        return {"subtask": subtask}

    retry_count = subtask.get("retry_count", 0)
    if retry_count < _sql_retry_max:
        subtask["retry_count"] = retry_count + 1
        # 保留 error，供 generate_sql_node 重试时使用
        subtask["status"] = "running"
        log.info(
            f"retry_decision: retry_count {retry_count}→{subtask['retry_count']}  "
            f"路由到 generate_sql（重试）  error={subtask.get('error', '-')}"
        )
    else:
        subtask["status"] = "failed"
        log.error(
            f"retry_decision: retry_count={retry_count} >= {_sql_retry_max}  "
            f"放弃重试，status=failed  error={subtask.get('error', '-')}"
        )

    return {"subtask": subtask}


# ============================================================
# 条件边: 重试判断后路由
# ============================================================
def _after_retry_decision(state: SubGraphState) -> Literal["done", "retry", "give_up"]:
    """retry_decision 节点后的条件边：按 status 映射到目标节点或 END。

    Returns:
        - "done": status == "success" → END
        - "retry": status == "running" → 回到 generate_sql
        - "give_up": status == "failed" → END
    """
    status = state["subtask"].get("status", "")
    if status == "success":
        return "done"
    elif status == "failed":
        return "give_up"
    else:
        return "retry"


# ============================================================
# 图编排: build_sql_subgraph
# ============================================================
def build_sql_subgraph() -> StateGraph:
    """构建 SQL 求解子图，返回 CompiledGraph。

    子图流程:
        generate_sql → validate_sql → [has_error?]
          ├── YES → retry_decision → [route]
          └── NO  → execute_sql    → retry_decision → [route]

        retry_decision 路由:
          - done    → END
          - retry   → generate_sql（重试）
          - give_up → END

    Returns:
        CompiledGraph: 已编译的 LangGraph 子图。
    """
    graph = StateGraph(SubGraphState)

    # 添加节点
    graph.add_node("generate_sql", _generate_sql_node)
    graph.add_node("validate_sql", _validate_sql_node)
    graph.add_node("execute_sql", _execute_sql_node)
    graph.add_node("retry_decision", _retry_decision_node)

    # 入口
    graph.set_entry_point("generate_sql")

    # 线性边: generate_sql → validate_sql
    graph.add_edge("generate_sql", "validate_sql")

    # 条件边: validate_sql → 无错误进 execute_sql，有错误进 retry_decision
    graph.add_conditional_edges(
        "validate_sql",
        _after_validate,
        {
            "execute_sql": "execute_sql",
            "retry_decision": "retry_decision",
        },
    )

    # 线性边: execute_sql → retry_decision
    graph.add_edge("execute_sql", "retry_decision")

    # 条件边: retry_decision → 三种出口
    graph.add_conditional_edges(
        "retry_decision",
        _after_retry_decision,
        {
            "done": END,
            "retry": "generate_sql",
            "give_up": END,
        },
    )

    return graph.compile()


# ============================================================
# 对外接口: solve_subtask
# ============================================================
def solve_subtask(
    subtask: SubTask,
    db_id: str,
    upstream_results: dict | None = None,
    trace_id: str | None = None,
) -> SubTask:
    """执行单个子任务的 SQL 求解全流程。

    1. 通过 get_schema 拉取目标数据库的 CREATE TABLE DDL。
    2. 构造 SubGraphState，invoke 子图。
    3. 返回填充了 sql / result / status / error 等字段的 subtask。

    Args:
        subtask: 待求解的子任务（需含 id, description, db_id 等字段）。
        db_id: 数据库标识符。
        upstream_results: 上游依赖子任务的结果映射 {dep_id: result}。
        trace_id: 日志追踪 ID，默认使用 subtask.id。

    Returns:
        SubTask: 完成求解后填充完整的子任务，包含 sql/result/status/error 等字段。
    """
    tid = trace_id or subtask.get("id", "-")
    log = get_logger(tid)

    # 拉取 Schema
    schema = get_schema(db_id)
    log.info(f"solve_subtask 开始  db_id={db_id}  schema长度={len(schema)}")

    # 构造初始状态
    initial_state: SubGraphState = {
        "subtask": subtask,
        "schema": schema,
        "db_id": db_id,
        "upstream_results": upstream_results or {},
        "_completed_tasks": [],
    }

    # 编译并 invoke 子图
    subgraph = build_sql_subgraph()
    t0 = time.perf_counter()
    final_state = subgraph.invoke(initial_state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result_subtask: SubTask = final_state["subtask"]
    log.info(
        f"solve_subtask 结束  总耗时 {elapsed_ms:.0f}ms  "
        f"status={result_subtask['status']}  "
        f"retry_count={result_subtask.get('retry_count', 0)}  "
        f"sql=[{result_subtask.get('sql', '')[:80] or '-'}]"
    )

    return result_subtask


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    from app.log_utils import new_trace_id

    tid = new_trace_id()
    log = get_logger(tid)
    log.info("=== sql_subgraph 自测开始 ===")

    # 手造 SubTask: 查询 department_store 库中 Customers 表的总数
    subtask: SubTask = {
        "id": f"subtask_{tid}",
        "description": "查询 Customers 表中客户的总数",
        "depends_on": [],
        "db_id": "department_store",
        "sql": None,
        "result": None,
        "status": "pending",
        "error": None,
        "retry_count": 0,
    }

    log.info(f"自测 SubTask: id={subtask['id']}  desc={subtask['description']}")

    try:
        result = solve_subtask(
            subtask=subtask,
            db_id="department_store",
            upstream_results=None,
            trace_id=tid,
        )
        log.info(f"=== 最终结果 ===")
        log.info(f"  status      = {result['status']}")
        log.info(f"  sql         = {result.get('sql', '-')}")
        log.info(f"  result      = {result.get('result', '-')}")
        log.info(f"  error       = {result.get('error', '-')}")
        log.info(f"  retry_count = {result.get('retry_count', '-')}")

        if result["status"] == "success":
            log.info("[PASS] sql_subgraph 自测通过")
        else:
            log.error(f"[FAIL] sql_subgraph 自测未达到预期: status={result['status']}")
    except Exception as e:
        log.error(f"[FAIL] sql_subgraph 自测异常: {e}")
        import traceback
        traceback.print_exc()

    log.info("=== sql_subgraph 自测结束 ===")
