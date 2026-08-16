"""
Text2SQL 批量评估模块。
逐条跑 build_main_graph，输出结构化指标，支持 API 模型 vs GRPO 模型对比。

用法:
    python evaluate.py                           # 默认用 eval_set.json，tag="default"
    python evaluate.py --tag grpo-v1             # 标注为 GRPO 微调模型
    python evaluate.py --set my_cases.json       # 自定义评估集
    python evaluate.py --db department_store       # 只看某个数据库
    python evaluate.py --timeout 180             # 单条超时（默认 120s）
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.db_utils import execute_sql
from app.graph import build_main_graph
from app.llm_client import call_json
from app.log_utils import get_logger, new_trace_id
from app.state import SubTask, init_main_state

# ============================================================
# 常量
# ============================================================
# 默认评估集（data/ 位于 backend/ 内，与运行时 traces.db 同目录）
_DEFAULT_EVAL_SET: str = str(_project_root / "data" / "eval_set.json")
_DEFAULT_TAG: str = "default"
_DEFAULT_TIMEOUT: int = 120
_DEFAULT_MAX_ITERATION: int = 3
_DEFAULT_OUTPUT_PREFIX: str = "eval_report"


# ============================================================
# 1. 评估数据加载
# ============================================================
def load_eval_set(filepath: str) -> list[dict[str, Any]]:
    """加载评估集 JSON 文件。

    每行格式:
        {
            "question": "…",      # 必填
            "db_id": "…",         # 必填
            "gold_sql": "…",      # 可选（至少与 gold_answer 二选一）
            "gold_answer": "…",   # 可选
        }

    Args:
        filepath: 评估集 JSON 文件路径。

    Returns:
        list[dict]: 评估用例列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: JSON 解析失败或结构不符合预期。
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"评估集文件不存在: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("评估集 JSON 顶层必须是数组")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 条评估用例不是 JSON 对象: {item}")
        if "question" not in item:
            raise ValueError(f"第 {i} 条评估用例缺少 question 字段")
        if "db_id" not in item:
            raise ValueError(f"第 {i} 条评估用例缺少 db_id 字段")
        if "gold_sql" not in item and "gold_answer" not in item:
            raise ValueError(
                f"第 {i} 条评估用例至少需要 gold_sql 或 gold_answer 之一"
            )

    return data


# ============================================================
# 2. SQL 结果集合比较 — 列序/行序无关
# ============================================================
def _normalize_result(result: dict) -> tuple[list[str], frozenset[tuple]] | tuple[None, None]:
    """将 execute_sql 返回的 dict 规范化为 (列名列表, 行集 frozenset)。

    保留列名列表用于后续列对齐；行放进 frozenset 消除行序。

    Args:
        result: execute_sql 返回的结构化 dict。

    Returns:
        tuple[list[str], frozenset] | (None, None): (列名列表, 行集)，失败时返回 (None, None)。
    """
    if not result or not result.get("success"):
        return None, None

    columns: list[str] = result.get("columns", [])
    rows: list[tuple] = result.get("rows", [])
    return columns, frozenset(tuple(row) for row in rows)


def _canonical_rows(columns: list[str], rows: frozenset[tuple]) -> frozenset[tuple]:
    """按列名排序重排每行，消除列序差异并锁定列身份。

    对列名做大小写不敏感排序得到列索引重排顺序，再把每行的值按该顺序重排。
    仅当两边列名集合一致时调用，此时同名列表现在同一位置，值互换能被识别。

    Args:
        columns: 列名列表（与行内值一一对应）。
        rows: 行集（每行按原列序存放）。

    Returns:
        frozenset[tuple]: 列按列名排序后的行集。
    """
    order = sorted(range(len(columns)), key=lambda i: columns[i].lower())
    return frozenset(tuple(row[i] for i in order) for row in rows)


# ============================================================
# 三级兜底: 值比对 / 无序集合比对 / LLM 判断
# ============================================================
_JUDGE_PROMPT = """你是一个 SQL 结果比对专家。请判断两个 SQL 查询结果是否表达了相同的数据含义（语义等价）。

## 原始问题
{question}

## 结果 A（Agent 生成）
列: {agent_cols}
数据: {agent_rows}

## 结果 B（标准答案 Gold）
列: {gold_cols}
数据: {gold_rows}

## 判断标准
- 忽略列名拼写差异（如 COUNT(*) 与 cnt 等价）、列顺序、行顺序。
- 比较两个结果表达的业务事实是否一致（数值是否相等、分组是否一致）。

请严格只输出一个 JSON 对象。"""


def _value_multiset(rows: frozenset[tuple]) -> Counter:
    """把结果所有单元格值收集为无序多重集合，忽略列名、列序与行序。

    Args:
        rows: 规范化后的行集（每行为 tuple）。

    Returns:
        Counter: 单元格值 → 出现次数的多重集合。
    """
    counter: Counter = Counter()
    for row in rows:
        counter.update(row)
    return counter


def _llm_judge_results(
    agent_result: dict,
    gold_result: dict,
    question: str | None,
    trace_id: str,
) -> tuple[bool, str]:
    """第三层兜底：确定性比较不一致时，调用 LLM 判断两个结果是否语义等价。

    Args:
        agent_result: Agent 子任务 SQL 的执行结果。
        gold_result: gold_sql 的执行结果。
        question: 原始问题（辅助语义判断，可为空）。
        trace_id: 日志追踪 ID。

    Returns:
        tuple[bool, str]: (是否等价, 判断说明)。
    """
    agent_cols = agent_result.get("columns", [])
    agent_rows = agent_result.get("rows", [])
    gold_cols = gold_result.get("columns", [])
    gold_rows = gold_result.get("rows", [])

    # 截断行数，避免 prompt 过长
    _max_disp_rows = 20
    agent_rows_disp = agent_rows[:_max_disp_rows]
    gold_rows_disp = gold_rows[:_max_disp_rows]
    if len(agent_rows) > _max_disp_rows:
        agent_rows_disp.append(f"…（共 {len(agent_rows)} 行，仅展示前 {_max_disp_rows} 行）")
    if len(gold_rows) > _max_disp_rows:
        gold_rows_disp.append(f"…（共 {len(gold_rows)} 行，仅展示前 {_max_disp_rows} 行）")

    prompt = _JUDGE_PROMPT.format(
        question=question or "（未提供）",
        agent_cols=agent_cols,
        agent_rows=agent_rows_disp,
        gold_cols=gold_cols,
        gold_rows=gold_rows_disp,
    )
    schema_hint = '{"equivalent": true/false, "reason": "等价或不等价的简要说明"}'

    try:
        judge = call_json(
            prompt=prompt,
            schema_hint=schema_hint,
            trace_id=trace_id,
            max_retry=1,
        )
    except ValueError as e:
        log = get_logger(trace_id)
        log.error(f"LLM 兜底判断失败: {e}")
        return False, f"确定性比较不一致，且 LLM 兜底判断失败（{type(e).__name__}）"

    equivalent = bool(judge.get("equivalent", False))
    reason = judge.get("reason", "")
    return equivalent, f"LLM 兜底: {'等价' if equivalent else '不等价'} — {reason}"


def compare_sql_results(
    agent_result: dict | None,
    gold_result: dict | None,
    question: str | None = None,
    trace_id: str | None = None,
) -> tuple[bool, str]:
    """比较 Agent 执行结果与 gold_sql 执行结果是否一致，三级兜底。

    三级比较策略：
        1. 字段一致 → 看值：两边列名集合一致时，按列名对齐后精确比较值。
        2. 字段不一致 → 看无序集合：列名不一致时，忽略列名/列序/行序，
           比较单元格值的多重集合。
        3. 仍不一致 → 大模型判断：前两层都不一致时，若提供 trace_id，
           调用 LLM 判断两个结果是否语义等价；未提供则直接判不一致。

    前置规则：
        - 两边都失败 → 一致（都是拿不到数据），返回 True
        - 一边失败一边成功 → 不一致

    Args:
        agent_result: Agent 子任务 SQL 的执行结果（db_utils.execute_sql 返回值）。
        gold_result: gold_sql 的执行结果。
        question: 原始问题（供 LLM 兜底判断，可选）。
        trace_id: 日志追踪 ID（提供时启用 LLM 兜底，可选）。

    Returns:
        tuple[bool, str]: (是否一致, 差异描述)。
    """
    agent_ok = agent_result and agent_result.get("success")
    gold_ok = gold_result and gold_result.get("success")

    # 两边都失败 → 视为一致（数据结构问题，非 SQL 问题）
    if not agent_ok and not gold_ok:
        return True, "两边均执行失败（可能是数据库结构问题）"

    if not agent_ok:
        return False, f"Agent SQL 执行失败: {agent_result.get('error', '未知')}"

    if not gold_ok:
        return False, f"gold_sql 执行失败（评估数据可能有问题）: {gold_result.get('error', '未知')}"

    # 规范化：拿到 (列名列表, 行集)，行序已消除
    agent_cols, agent_rows = _normalize_result(agent_result)
    gold_cols, gold_rows = _normalize_result(gold_result)

    if agent_rows is None or gold_rows is None:
        return False, "结果规范化失败"

    # ---- 第 1 层：字段一致 → 看值（按列名对齐后精确比较） ----
    if {c.lower() for c in agent_cols} == {c.lower() for c in gold_cols}:
        agent_rows_aligned = _canonical_rows(agent_cols, agent_rows)
        gold_rows_aligned = _canonical_rows(gold_cols, gold_rows)
        if agent_rows_aligned == gold_rows_aligned:
            return True, "字段一致，值完全一致"

    # ---- 第 2 层：字段不一致 → 看无序集合 ----
    else:
        if _value_multiset(agent_rows) == _value_multiset(gold_rows):
            return True, "字段不一致，但值无序集合一致"

    # ---- 第 3 层：还不一致 → 大模型判断 ----
    if trace_id:
        return _llm_judge_results(agent_result, gold_result, question, trace_id)

    return False, "字段与值均不一致（未启用 LLM 兜底）"


# ============================================================
# 2.5 结果可视化 — 生成/原始 SQL 执行结果对比
# ============================================================
def _format_result_block(result: dict | None, max_rows: int = 20) -> str:
    """将 execute_sql 返回的结果 dict 渲染成可读文本块，便于人工对比。

    Args:
        result: execute_sql 返回的结构化 dict（或 None）。
        max_rows: 最多展示的数据行数，防止结果过长刷屏。

    Returns:
        str: 格式化后的文本块。
    """
    if not result:
        return "（无执行结果）"

    if not result.get("success"):
        return f"执行失败: {result.get('error', '未知')}"

    columns: list[str] = result.get("columns", [])
    rows: list[tuple] = result.get("rows", [])

    lines: list[str] = [
        f"列名({len(columns)}): {', '.join(columns)}",
        f"行数: {len(rows)}",
    ]

    # 有数据时渲染成简单对齐的表格
    if columns and rows:
        lines.append("  " + " | ".join(columns))
        lines.append("  " + " | ".join(["---"] * len(columns)))
        for row in rows[:max_rows]:
            lines.append("  " + " | ".join(str(c) for c in row))
        if len(rows) > max_rows:
            lines.append(f"  …（仅展示前 {max_rows} 行，共 {len(rows)} 行）")
    return "\n".join(lines)


def format_sql_diff(eval_result: dict[str, Any], max_rows: int = 20) -> str:
    """生成「生成的 SQL vs 原始 SQL」执行结果对比文本，便于定位差异。

    依次展示：
        1. gold_sql（原始 SQL）及其执行结果。
        2. 每个 Agent 子任务的 SQL、执行结果、与 gold 的比对结论。

    Args:
        eval_result: evaluate_one 返回的单条评估结果。
        max_rows: 每个结果最多展示的数据行数。

    Returns:
        str: 对比文本块。
    """
    sep = "─" * 60
    lines: list[str] = [sep]

    # ---- 原始 SQL 侧 ----
    lines.append("【原始 SQL (gold_sql)】")
    lines.append(eval_result.get("gold_sql") or "（无）")
    lines.append("— 执行结果 —")
    lines.append(_format_result_block(eval_result.get("gold_result"), max_rows))
    lines.append(sep)

    # ---- 生成的 SQL 侧 ----
    subtasks = eval_result.get("subtask_details", [])
    if not subtasks:
        lines.append("【生成的 SQL (Agent)】")
        lines.append("（无子任务详情）")
        lines.append(sep)
    else:
        for detail in subtasks:
            lines.append(
                f"【生成的 SQL (Agent · 子任务 {detail.get('id')}) · status={detail.get('status')}】"
            )
            lines.append(detail.get("sql") or "（无）")
            lines.append("— 执行结果 —")
            lines.append(_format_result_block(detail.get("result"), max_rows))
            if detail.get("error"):
                lines.append(f"[错误] {detail['error']}")
            if detail.get("sql_match_gold") is not None:
                mark = "✓ 一致" if detail["sql_match_gold"] else "✗ 不一致"
                lines.append(f"[比对] {mark}: {detail.get('sql_match_detail', '')}")
            lines.append(sep)

    return "\n".join(lines)


# ============================================================
# 3. 单条评估执行
# ============================================================
def evaluate_one(
    case: dict[str, Any],
    max_iteration: int = _DEFAULT_MAX_ITERATION,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """执行单条评估用例，返回结构化评估结果。

    Args:
        case: 评估用例 dict，含 question/db_id/gold_sql(可选)/gold_answer(可选)。
        max_iteration: 最大反思轮次，默认 3。
        timeout: 单条超时秒数，默认 120。

    Returns:
        dict: 评估结果，包含所有指标字段。
    """
    trace_id: str = new_trace_id()
    log = get_logger(trace_id)
    question: str = case["question"]
    db_id: str = case["db_id"]
    gold_sql: str | None = case.get("gold_sql")

    result: dict[str, Any] = {
        "trace_id": trace_id,
        "question": question,
        "db_id": db_id,
        "gold_sql": gold_sql,
        "status": "error",          # done | degraded | timeout | error
        "final_answer": None,
        "reflection_passed": None,  # 首轮反思是否通过
        "iterations": 0,            # 实际迭代轮次
        "subtask_count": 0,         # 拆解出的子任务数
        "subtask_details": [],      # 每个子任务的 SQL + 执行结果
        "sql_accuracy": None,       # SQL 执行结果是否与 gold_sql 一致
        "sql_accuracy_detail": "",  # 差异说明
        "gold_result": None,        # gold_sql 的完整执行结果（columns + rows），便于人工对比
        "elapsed_ms": 0,
        "error": None,
    }

    t_start = time.perf_counter()

    try:
        # 构建图并在线程池中执行（带超时）
        graph = build_main_graph()
        state = init_main_state(
            question=question,
            db_id=db_id,
            trace_id=trace_id,
            max_iteration=max_iteration,
        )

        def _run() -> dict[str, Any]:
            """在线程中同步 invoke 图。"""
            return graph.invoke(state)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                final_state = future.result(timeout=timeout)
            except FutureTimeoutError:
                result["status"] = "timeout"
                result["error"] = f"单条超时（>{timeout}s）"
                result["elapsed_ms"] = (time.perf_counter() - t_start) * 1000
                log.error(f"evaluate_one 超时  timeout={timeout}s")
                return result

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result["elapsed_ms"] = elapsed_ms

        # 提取状态字段
        final_status: str = final_state.get("status", "error")
        result["status"] = final_status
        result["final_answer"] = final_state.get("final_answer")
        result["iterations"] = final_state.get("iteration", 0)
        plan: list[SubTask] = final_state.get("plan", [])
        result["subtask_count"] = len(plan)

        # 反思是否通过
        reflection = final_state.get("reflection")
        if isinstance(reflection, dict):
            result["reflection_passed"] = reflection.get("passed")
            if not reflection.get("passed"):
                result["reflection_reason"] = reflection.get("reason", "")

        # 收集子任务详情
        completed: dict[str, SubTask] = final_state.get("completed", {})
        for sid, task in completed.items():
            detail: dict[str, Any] = {
                "id": sid,
                "description": task.get("description", ""),
                "status": task.get("status", ""),
                "sql": task.get("sql"),
                "result": task.get("result"),
                "error": task.get("error"),
            }
            result["subtask_details"].append(detail)

        # ---- SQL 执行准确率评估 ----
        if gold_sql:
            # 执行 gold_sql
            gold_result = execute_sql(db_id=db_id, sql=gold_sql)
            result["gold_result"] = gold_result  # 持久化，供对比展示与报告
            if not gold_result.get("success"):
                result["sql_accuracy_detail"] = (
                    f"gold_sql 执行失败: {gold_result.get('error', '未知')}"
                )
                result["sql_accuracy"] = None  # 无法比较
            else:
                # 合并所有成功子任务的 SQL 结果与 gold 对比
                # 策略：如果有多个子任务，尝试找匹配 gold 的那个（通过结果集对比）
                best_match = False
                best_detail = ""
                fail_detail = ""  # 记录最后一条不匹配的原因
                for detail_item in result["subtask_details"]:
                    if detail_item["status"] != "success":
                        continue
                    sub_result = detail_item.get("result")
                    if not sub_result:
                        continue
                    is_match, desc = compare_sql_results(
                        sub_result, gold_result, question=question, trace_id=trace_id
                    )
                    detail_item["sql_match_gold"] = is_match
                    detail_item["sql_match_detail"] = desc
                    if is_match:
                        best_match = True
                        best_detail = desc
                    else:
                        fail_detail = desc

                result["sql_accuracy"] = best_match
                if best_match:
                    result["sql_accuracy_detail"] = best_detail
                elif fail_detail:
                    result["sql_accuracy_detail"] = fail_detail
                else:
                    result["sql_accuracy_detail"] = "未找到成功子任务可对比"

        log.info(
            f"evaluate_one 完成  status={final_status}  "
            f"elapsed={elapsed_ms:.0f}ms  sql_accuracy={result['sql_accuracy']}"
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_ms"] = (time.perf_counter() - t_start) * 1000
        log.error(f"evaluate_one 异常: {result['error']}")

    return result


# ============================================================
# 4. 汇总计算
# ============================================================
def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """根据逐条评估结果计算汇总指标。

    Args:
        results: evaluate_one 返回的逐条结果列表。

    Returns:
        dict: 汇总指标。
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "error": "无评估结果"}

    # 成功完成（非 timeout 非 error）
    completed = [r for r in results if r["status"] not in ("timeout", "error")]
    # 状态为 done 的（反思通过）
    done = [r for r in results if r["status"] == "done"]
    # 首轮反思通过
    first_pass = [
        r for r in results
        if r.get("reflection_passed") is True and r.get("iterations", 0) == 0
    ]
    # 有 sql_accuracy 结果的
    sql_evaluable = [
        r for r in results
        if r.get("sql_accuracy") is not None and r["status"] not in ("timeout", "error")
    ]
    sql_correct = [r for r in sql_evaluable if r["sql_accuracy"] is True]

    # 平均耗时（仅成功完成）
    elapsed_list = [r["elapsed_ms"] for r in completed if r.get("elapsed_ms", 0) > 0]

    return {
        "total": total,
        "completed_count": len(completed),
        "done_count": len(done),
        "degraded_count": len([r for r in results if r["status"] == "degraded"]),
        "timeout_count": len([r for r in results if r["status"] == "timeout"]),
        "error_count": len([r for r in results if r["status"] == "error"]),
        # 一次成功率（首轮反思即通过 / 总完成数）
        "first_pass_rate": (
            len(first_pass) / len(completed) if completed else 0.0
        ),
        "first_pass_count": len(first_pass),
        # 最终成功率（done / total）
        "final_success_rate": len(done) / total if total > 0 else 0.0,
        # 平均迭代轮数
        "avg_iterations": (
            sum(r["iterations"] for r in results) / total if total > 0 else 0.0
        ),
        # SQL 执行准确率
        "sql_accuracy": (
            len(sql_correct) / len(sql_evaluable) if sql_evaluable else None
        ),
        "sql_correct_count": len(sql_correct),
        "sql_evaluable_count": len(sql_evaluable),
        # 平均耗时（ms）
        "avg_elapsed_ms": (
            sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0
        ),
        "min_elapsed_ms": min(elapsed_list) if elapsed_list else 0.0,
        "max_elapsed_ms": max(elapsed_list) if elapsed_list else 0.0,
    }


# ============================================================
# 5. 报告输出
# ============================================================
def format_table(metrics: dict[str, Any], tag: str) -> str:
    """生成可打印的汇总表格。

    Args:
        metrics: compute_metrics 返回的汇总指标。
        tag: 模型标签。

    Returns:
        str: 格式化的汇总表格字符串。
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  Text2SQL 评估报告 — 模型: {tag}")
    lines.append("=" * 70)
    lines.append("")

    # 基础统计
    lines.append(f"  总用例数:           {metrics['total']}")
    lines.append(f"  成功完成:           {metrics['completed_count']}  "
                 f"(done={metrics['done_count']}, degraded={metrics['degraded_count']})")
    lines.append(f"  超时:               {metrics['timeout_count']}")
    lines.append(f"  异常:               {metrics['error_count']}")
    lines.append("")

    # 成功率
    lines.append(f"  一次成功率:         {metrics['first_pass_rate']:.1%}  "
                 f"({metrics['first_pass_count']}/{metrics['completed_count']})")
    lines.append(f"  最终成功率:         {metrics['final_success_rate']:.1%}  "
                 f"({metrics['done_count']}/{metrics['total']})")
    lines.append(f"  平均迭代轮数:       {metrics['avg_iterations']:.2f}")
    lines.append("")

    # SQL 准确率
    if metrics["sql_accuracy"] is not None:
        lines.append(
            f"  SQL 执行准确率:     {metrics['sql_accuracy']:.1%}  "
            f"({metrics['sql_correct_count']}/{metrics['sql_evaluable_count']})"
        )
    else:
        lines.append("  SQL 执行准确率:     N/A（无 gold_sql 或全部无法比较）")
        lines.append(f"    可评估 SQL 条数:  {metrics['sql_evaluable_count']}")
    lines.append("")

    # 耗时
    lines.append(f"  平均耗时:           {metrics['avg_elapsed_ms']:.0f} ms")
    lines.append(f"  最小时耗:           {metrics['min_elapsed_ms']:.0f} ms")
    lines.append(f"  最大耗时:           {metrics['max_elapsed_ms']:.0f} ms")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_report(
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
    tag: str,
    output_dir: str | None = None,
) -> str:
    """将评估结果和指标写入 JSON 文件。

    Args:
        results: 逐条评估结果。
        metrics: 汇总指标。
        tag: 模型标签。
        output_dir: 输出目录，默认当前目录。

    Returns:
        str: 输出文件路径。
    """
    out_dir = Path(output_dir) if output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{_DEFAULT_OUTPUT_PREFIX}_{tag}.json"

    report = {
        "meta": {
            "tag": tag,
            "total": metrics["total"],
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        },
        "metrics": metrics,
        "details": results,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    return str(filename.resolve())


# ============================================================
# 6. 主入口
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text2SQL Agent 批量评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python evaluate.py                              # 默认评估\n"
            '  python evaluate.py --tag grpo-v1               # 标注 GRPO 模型\n'
            '  python evaluate.py --set my_cases.json          # 自定义评估集\n'
            '  python evaluate.py --db department_store          # 按数据库过滤\n'
            "  python evaluate.py --timeout 180 --max-iter 2   # 自定义参数"
        ),
    )
    parser.add_argument(
        "--set", dest="eval_set",
        type=str,
        default=_DEFAULT_EVAL_SET,
        help=f"评估集 JSON 文件路径（默认: {_DEFAULT_EVAL_SET}）",
    )
    parser.add_argument(
        "--tag", "-t",
        type=str,
        default=_DEFAULT_TAG,
        help=f"模型标签，用于标注评估轮次（默认: {_DEFAULT_TAG}）",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="仅评估指定 db_id 的用例",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT,
        help=f"单条超时秒数（默认: {_DEFAULT_TIMEOUT}）",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=_DEFAULT_MAX_ITERATION,
        help=f"最大反思轮次（默认: {_DEFAULT_MAX_ITERATION}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="报告输出目录（默认当前目录）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印每条评估的详细信息",
    )

    args = parser.parse_args()

    # 加载评估集
    print(f"加载评估集: {args.eval_set}")
    cases = load_eval_set(args.eval_set)
    print(f"  共 {len(cases)} 条用例")

    # 按 db_id 过滤
    if args.db:
        before = len(cases)
        cases = [c for c in cases if c["db_id"] == args.db]
        print(f"  按 db_id={args.db!r} 过滤后: {len(cases)} 条（原 {before} 条）")

    if not cases:
        print("  无可用评估用例，退出。")
        sys.exit(0)

    # 逐条评估
    print(f"\n开始评估（tag={args.tag}, timeout={args.timeout}s, max_iter={args.max_iter}）")
    print("-" * 60)

    results: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        q_preview = case["question"][:60] + (
            "…" if len(case["question"]) > 60 else ""
        )
        print(f"[{i}/{len(cases)}] {q_preview}  ", end="", flush=True)

        eval_result = evaluate_one(
            case,
            max_iteration=args.max_iter,
            timeout=args.timeout,
        )
        results.append(eval_result)

        # 单条状态摘要
        status = eval_result["status"]
        elapsed = eval_result["elapsed_ms"]
        sql_acc = eval_result.get("sql_accuracy")
        sql_str = ""
        if sql_acc is True:
            sql_str = "  SQL=PASS"
        elif sql_acc is False:
            sql_str = "  SQL=FAIL"
        elif sql_acc is None:
            sql_str = "  SQL=N/A"

        print(f"→ {status}  {elapsed:.0f}ms  iter={eval_result['iterations']}{sql_str}")

        if args.verbose and eval_result.get("error"):
            print(f"    错误: {eval_result['error']}")
        if args.verbose and eval_result.get("sql_accuracy_detail"):
            print(f"    SQL详情: {eval_result['sql_accuracy_detail']}")
        # 展示 SQL 执行结果对比：比对失败时总是打印（方便定位问题），verbose 时始终打印
        if sql_acc is False or args.verbose:
            print(format_sql_diff(eval_result))

    # 汇总
    metrics = compute_metrics(results)
    table = format_table(metrics, args.tag)
    print(table)

    # 保存报告
    report_path = save_report(results, metrics, args.tag, args.output_dir)
    print(f"\n报告已保存: {report_path}")

    # 返回码：全部成功 → 0，有失败 → 1
    failed = metrics["error_count"] + metrics["timeout_count"]
    if failed > 0:
        print(f"[WARN] {failed} 条用例未正常完成（超时/异常）")
    if metrics["sql_accuracy"] is not None and metrics["sql_accuracy"] < 1.0:
        print(f"[WARN] SQL 准确率 < 100%，详见报告文件中的 subtask_details.sql_match_gold 字段")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
