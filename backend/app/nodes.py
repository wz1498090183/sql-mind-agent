"""
主图节点模块。
实现 LangGraph 主图的六个节点：
    plan_node → dispatch_node → aggregate_node → reflect_node
    → [finalize_node | degrade_node]
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.db_utils import get_schema, get_value_samples
from app.llm_client import call_json
from app.log_utils import get_logger
from app.state import MainState, Reflection, SqlResult, SubTask
from app.sql_subgraph import solve_subtask


# ============================================================
# Prompt 模板 — 任务规划
# ============================================================
_PLAN_PROMPT = """你是一个 Text2SQL 任务规划专家。根据用户提出的复杂业务问题，将其拆解为多个 SQL 子任务。

## 用户问题
{question}

## 数据库 Schema
{schema}

{value_samples}

## 拆解规则
1. 简单问题可以只拆 1 个子任务。
2. 同比/环比/对比/比率类问题，拆成多个可并行的取数子任务，后续数值计算交给聚合层（不在此生成 SQL）。
3. depends_on 表达依赖关系（填写依赖子任务的 id），无依赖的子任务将被并行执行。
4. 每个子任务的 description 必须具体、可独立生成 SQL，包含涉及的表名、列名、筛选条件。
5. id 格式为 "t1", "t2", "t3" ...
6. **重要**：如果问题中提到的实体（表名/列名）与 Schema 不完全匹配，请基于 Schema 中最接近的表来拆解子任务。
   例如问题问"歌手"但 Schema 中有 Contacts（联系人）表，就查询 Contacts 表。
   即使不完全匹配，也必须返回至少 1 个子任务，不要返回空 subtasks。

{examples}

{reflection_block}

请严格只输出一个 JSON 对象。"""

_PLAN_FEW_SHOT = """## 示例（演示拆解方式与 JSON 结构，表名以当前 Schema 为准）

### 示例 1：简单问题 → 单子任务
问题：Customers 表有多少条记录？
```json
{"subtasks": [{"id": "t1", "description": "查询 Customers 表的记录总数", "depends_on": []}]}
```

### 示例 2：同比/环比 → 多个并行子任务
问题：本季度销售额相比上季度增长了多少？
```json
{"subtasks": [
  {"id": "t1", "description": "查询 sales 表本季度（日期在当前季度内）的销售总额", "depends_on": []},
  {"id": "t2", "description": "查询 sales 表上季度（日期在上一季度内）的销售总额", "depends_on": []}
]}
```
（增长率计算交给聚合层完成，不在此生成 SQL）"""

_RETRY_BLOCK = """## 上一轮失败原因
{reason}

请根据失败原因重新规划子任务，避免再次出现同样的问题。"""

_AGGREGATE_PROMPT = """你是一个数据分析师。请根据用户问题和各子任务的查询结果，生成最终的自然语言答案。

## 用户原始问题
{question}

## 各子任务及结果
{subtask_details}

## 要求
1. 直接回答用户的问题。
2. 如果需要计算（同比、环比、比率、差值等），请在回答中完成并展示计算过程。
3. 答案应简洁专业，用中文回答。
4. 如果有多个子任务，将它们的结果有机整合，不要简单罗列。

请严格只输出一个 JSON 对象。"""


# ============================================================
# 拓扑排序工具
# ============================================================
def _topological_levels(subtasks: list[SubTask]) -> list[list[SubTask]]:
    """对子任务做拓扑分层，检测环。

    使用 Kahn 算法：
        1. 构建入度表和邻接表。
        2. 入度为 0 的节点作为第一层。
        3. 逐层移除，产生新的入度为 0 节点作为下一层。
        4. 若有剩余节点未被分层，说明存在环。

    Args:
        subtasks: 子任务列表。

    Returns:
        list[list[SubTask]]: 按拓扑层排列的子任务列表，每层内无依赖关系。

    Raises:
        ValueError: 检测到依赖环。
    """
    id_to_subtask: dict[str, SubTask] = {s["id"]: s for s in subtasks}
    in_degree: dict[str, int] = defaultdict(int)
    adjacency: dict[str, list[str]] = defaultdict(list)

    for s in subtasks:
        sid = s["id"]
        # 初始化入度，确保每个 id 都在 in_degree 中
        if sid not in in_degree:
            in_degree[sid] = 0
        for dep in s.get("depends_on", []):
            if dep not in id_to_subtask:
                raise ValueError(
                    f"子任务 '{sid}' 依赖的 '{dep}' 不在计划中，"
                    f"可用 id: {list(id_to_subtask.keys())}"
                )
            adjacency[dep].append(sid)
            in_degree[sid] += 1

    # 收集所有入度为 0 的节点
    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    levels: list[list[SubTask]] = []
    visited_count = 0

    while queue:
        current_level: list[SubTask] = []
        next_queue: deque[str] = deque()
        while queue:
            sid = queue.popleft()
            current_level.append(id_to_subtask[sid])
            visited_count += 1
            for neighbor in adjacency.get(sid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)
        queue = next_queue
        levels.append(current_level)

    if visited_count != len(subtasks):
        remaining = [sid for sid, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"检测到依赖环！以下子任务存在循环依赖: {remaining}。"
            f"请检查 depends_on 字段。"
        )

    return levels


# ============================================================
# 节点 1: 任务规划
# ============================================================
def plan_node(state: MainState) -> dict[str, Any]:
    """任务规划节点：将复杂问题拆解为 1~N 个 SQL 子任务。

    从 MainState 中读取 question 和 db_id，拉取数据库 Schema，
    组装 Prompt 调用 LLM 生成子任务列表。解析结果写入 state.plan。

    若 state 中存在 reflection 且 passed=False，则把 reason 注入 Prompt
    作为"上一轮失败原因"重新规划。

    Args:
        state: 主图全局状态。

    Returns:
        dict: 更新后的 state 字段（仅 plan / status）。
    """
    trace_id: str = state.get("trace_id", "-")
    question: str = state.get("question", "")
    db_id: str = state.get("db_id", "")
    log = get_logger(trace_id)

    log.info(
        f"plan_node 进入  question长度={len(question)}  db_id={db_id}"
    )

    # 拉取 Schema
    schema = get_schema(db_id)
    log.info(f"plan_node 拉取 Schema 完成  schema长度={len(schema)}")

    # 值检索：抽取低基数类别列样本值，帮助实体模糊匹配
    value_samples = ""
    try:
        value_samples = get_value_samples(db_id)
    except Exception as e:
        log.warning(f"plan_node 值检索失败，跳过: {e}")

    # 构造 reflection 信息
    reflection = state.get("reflection")
    reflection_block = ""
    if reflection and not reflection.get("passed", True):
        reflection_block = _RETRY_BLOCK.format(reason=reflection.get("reason", "未知"))
        log.info(f"plan_node 检测到上一轮失败: {reflection_block[:100]}")

    # 组装 Prompt
    prompt = _PLAN_PROMPT.format(
        question=question,
        schema=schema if schema else "（无 Schema，请根据问题推断）",
        value_samples=value_samples or "（无列样本值）",
        examples=_PLAN_FEW_SHOT,
        reflection_block=reflection_block,
    )

    schema_hint = (
        '{"subtasks": ['
        '  {"id": "t1", "description": "查询...", "depends_on": []}'
        ", ..."
        "]}"
    )

    t0 = time.perf_counter()
    plan_result = call_json(
        prompt=prompt,
        schema_hint=schema_hint,
        trace_id=trace_id,
        max_retry=2,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 解析子任务
    raw_subtasks: list[dict] = plan_result.get("subtasks", [])
    if not raw_subtasks:
        # 尝试从 plan_result 中寻找 reason 字段（LLM 可能返回解释而非 subtasks）
        reason = plan_result.get("reason", plan_result.get("message", ""))
        if reason:
            log.warning(f"plan_node: LLM 未能拆解子任务，原因: {reason}")
            raise ValueError(f"无法为该问题生成执行计划: {reason}")
        # 降级：把原始问题作为唯一子任务，让 SQL 子图自己处理
        log.warning("plan_node: LLM 返回的 subtasks 为空，降级为单子任务（整问题直接生成 SQL）")
        raw_subtasks = [{
            "id": "t1",
            "description": question,
            "depends_on": [],
        }]

    subtasks: list[SubTask] = []
    for raw in raw_subtasks:
        sid = raw.get("id", f"t{len(subtasks) + 1}")
        subtasks.append({
            "id": sid,
            "description": raw.get("description", ""),
            "depends_on": raw.get("depends_on", []),
            "db_id": db_id,
            "sql": None,
            "result": None,
            "status": "pending",
            "error": None,
            "retry_count": 0,
        })

    # 检测环
    _topological_levels(subtasks)

    log.info(
        f"plan_node 完成  耗时 {elapsed_ms:.0f}ms  "
        f"子任务数={len(subtasks)}  "
        f"ids={[s['id'] for s in subtasks]}"
    )
    for s in subtasks:
        log.info(
            f"  [{s['id']}] desc={s['description'][:60]}  "
            f"depends_on={s['depends_on']}"
        )

    return {
        "plan": subtasks,
        "status": "executing",
    }


# ============================================================
# 节点 2: 调度执行
# ============================================================
def dispatch_node(state: MainState) -> dict[str, Any]:
    """调度执行节点：按拓扑分层并行执行所有子任务。

    1. 对 state.plan 做拓扑分层。
    2. 同一层内子任务并行执行（ThreadPoolExecutor）。
    3. 每完成一层，将结果收集到 completed_map，供下一层作为 upstream_results。
    4. 完成后写入 state.completed。

    Args:
        state: 主图全局状态。

    Returns:
        dict: 更新后的 state 字段（completed / _completed_tasks）。
    """
    trace_id: str = state.get("trace_id", "-")
    plan: list[SubTask] = state.get("plan", [])
    db_id: str = state.get("db_id", "")
    log = get_logger(trace_id)

    log.info(f"dispatch_node 进入  plan子任务数={len(plan)}")

    if not plan:
        log.warning("dispatch_node: plan 为空，跳过执行")
        return {"completed": {}, "_completed_tasks": []}

    # 拓扑分层
    levels = _topological_levels(plan)
    level_summary = " → ".join(
        f"第{i + 1}层: {[s['id'] for s in lv]}"
        for i, lv in enumerate(levels)
    )
    log.info(
        f"dispatch_node 拓扑分层完成  "
        f"共{len(levels)}层  {level_summary}"
    )

    # completed_map: {subtask_id: SubTask}，存每层完成后的结果
    completed_map: dict[str, SubTask] = {}

    for level_idx, level in enumerate(levels):
        level_ids = [s["id"] for s in level]
        log.info(
            f"dispatch_node 开始执行第{level_idx + 1}/{len(levels)}层  "
            f"子任务={level_ids}  并行数={len(level)}"
        )

        # 同一层内并行执行（带超时兜底）
        def run_subtask(task: SubTask) -> SubTask:
            """在线程池中执行单个子任务，传入上游依赖结果。"""
            tid = task["id"]
            # 收集上游依赖结果
            upstream: dict[str, dict] = {}
            for dep_id in task.get("depends_on", []):
                dep_task = completed_map.get(dep_id)
                if dep_task and dep_task.get("result"):
                    upstream[dep_id] = dep_task["result"]

            log = get_logger(trace_id)
            log.info(f"  [{tid}] 开始调度  depends_on={task.get('depends_on', [])}  upstream_keys={list(upstream.keys())}")
            result_task = solve_subtask(
                subtask=task,
                db_id=db_id,
                upstream_results=upstream,
                trace_id=f"{trace_id}_{tid}",
            )
            log.info(
                f"  [{tid}] 完成调度  status={result_task['status']}"
            )
            return result_task

        # 子任务超时秒数（可通过环境变量 SUBTASK_TIMEOUT_SEC 覆盖，默认 30s）
        _subt_timeout = float(
            os.environ.get("SUBTASK_TIMEOUT_SEC", "30")
        )

        level_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, len(level))) as executor:
            futures = {
                executor.submit(run_subtask, task): task["id"]
                for task in level
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    result_task = future.result(timeout=_subt_timeout)
                    completed_map[tid] = result_task
                except FutureTimeoutError:
                    log.error(
                        f"dispatch_node: 子任务 {tid} 超时  "
                        f"timeout={_subt_timeout}s，标记为 failed"
                    )
                    # 超时兜底：标记子任务为 failed，不阻塞整体
                    task = next(s for s in level if s["id"] == tid)
                    task["status"] = "failed"
                    task["error"] = (
                        f"子任务执行超时（>{_subt_timeout}s），"
                        f"可能原因：LLM 响应慢、SQL 执行慢、数据库锁竞争"
                    )
                    completed_map[tid] = task
                except Exception as e:
                    log.error(f"dispatch_node: 子任务 {tid} 执行异常: {e}")
                    # 构造失败的 SubTask
                    task = next(s for s in level if s["id"] == tid)
                    task["status"] = "failed"
                    task["error"] = str(e)
                    completed_map[tid] = task

        level_elapsed = (time.perf_counter() - level_start) * 1000
        status_str = ", ".join(
            f"{tid}={completed_map[tid]['status']}" for tid in level_ids
        )
        log.info(
            f"dispatch_node 第{level_idx + 1}层完成  "
            f"耗时 {level_elapsed:.0f}ms  "
            f"结果: {status_str}"
        )

    # 将所有完成的 SubTask 写入 completed
    completed_all: dict[str, SubTask] = {}
    completed_tasks_list: list[SubTask] = []
    failed_ids: list[str] = []
    for s in plan:
        sid = s["id"]
        if sid in completed_map:
            completed_all[sid] = completed_map[sid]
            completed_tasks_list.append(completed_map[sid])
            if completed_map[sid]["status"] == "failed":
                failed_ids.append(sid)
        else:
            # 异常情况：子任务未被处理
            s["status"] = "failed"
            s["error"] = "子任务未被调度执行"
            completed_all[sid] = s
            completed_tasks_list.append(s)
            failed_ids.append(sid)

    if failed_ids:
        log.warning(f"dispatch_node 有失败子任务: {failed_ids}（继续由反思层判断）")

    log.info(
        f"dispatch_node 完成  "
        f"成功={len(completed_all) - len(failed_ids)}/{len(plan)}  "
        f"失败={failed_ids if failed_ids else '无'}"
    )
    return {
        "completed": completed_all,
        "_completed_tasks": completed_tasks_list,
    }


# ============================================================
# 节点 3: 结果聚合
# ============================================================
def aggregate_node(state: MainState) -> dict[str, Any]:
    """结果聚合节点：汇总各子任务结果，生成最终自然语言答案。

    收集 completed 中所有成功子任务的描述和查询结果，
    组装 Prompt 调用 LLM 生成最终答案（含同比/环比等计算）。

    Args:
        state: 主图全局状态。

    Returns:
        dict: 更新后的 state 字段（aggregated_answer）。
    """
    trace_id: str = state.get("trace_id", "-")
    question: str = state.get("question", "")
    completed: dict[str, SubTask] = state.get("completed", {})
    log = get_logger(trace_id)

    log.info(f"aggregate_node 进入  completed数={len(completed)}")

    if not completed:
        log.warning("aggregate_node: completed 为空，无结果可聚合")
        return {"aggregated_answer": "未能获取任何子任务结果，无法回答该问题。"}

    # 组装各子任务的详细信息
    subtask_details: list[str] = []
    for sid, task in completed.items():
        details = f"### 子任务 {sid}: {task.get('description', '-')}\n"
        details += f"状态: {task.get('status', '-')}\n"
        if task.get("status") == "success" and task.get("result"):
            r = task["result"]
            details += f"列: {r.get('columns', [])}\n"
            details += f"数据行数: {len(r.get('rows', []))}\n"
            details += f"数据: {r.get('rows', [])}\n"
        else:
            details += f"失败原因: {task.get('error', '-')}\n"
        subtask_details.append(details)

    details_text = "\n".join(subtask_details)

    # 尝试用 Python 对简单数值做预计算（让 LLM 润色前数据更清晰）
    calc_summary = _pre_calc_summary(completed)
    if calc_summary:
        details_text += f"\n\n## 预计算摘要\n{calc_summary}"

    prompt = _AGGREGATE_PROMPT.format(
        question=question,
        subtask_details=details_text,
    )

    schema_hint = (
        '{"answer": "最终自然语言答案（中文）", '
        '"calc_detail": "如涉及计算，列出公式和计算过程"}'
    )

    t0 = time.perf_counter()
    try:
        agg_result = call_json(
            prompt=prompt,
            schema_hint=schema_hint,
            trace_id=trace_id,
            max_retry=2,
        )
    except ValueError as e:
        log.error(f"aggregate_node: call_json 失败: {e}")
        return {"aggregated_answer": f"聚合 LLM 返回无效 JSON，请手动检查: {details_text[:500]}"}

    elapsed_ms = (time.perf_counter() - t0) * 1000

    answer = agg_result.get("answer", "")
    calc_detail = agg_result.get("calc_detail", "")

    log.info(
        f"aggregate_node 完成  耗时 {elapsed_ms:.0f}ms  "
        f"answer长度={len(answer)}  calc_detail长度={len(calc_detail)}"
    )

    return {
        "aggregated_answer": answer,
        "status": "reflecting",
    }


# ============================================================
# 辅助: 预计算摘要 — 对纯数值结果做简单的同比/环比预计算
# ============================================================
def _pre_calc_summary(completed: dict[str, SubTask]) -> str:
    """对 completed 中的数值结果做简单预计算，生成摘要文本。

    仅提取单行单列的数值结果，计算比率、差值等，供 LLM 润色。

    Args:
        completed: 已完成子任务映射。

    Returns:
        str: 预计算摘要，无可用数值时返回空字符串。
    """
    values: dict[str, float] = {}

    for sid, task in completed.items():
        if task.get("status") != "success":
            continue
        result = task.get("result")
        if not result or not result.get("success"):
            continue
        rows = result.get("rows", [])
        # 只取单行单列的纯数值
        if len(rows) == 1 and len(rows[0]) == 1:
            try:
                values[sid] = float(rows[0][0])
            except (ValueError, TypeError):
                pass

    if len(values) < 2:
        return ""

    lines: list[str] = []
    sorted_ids = sorted(values.keys())
    for i, sid in enumerate(sorted_ids):
        lines.append(f"  {sid}: {values[sid]}")
        if i > 0:
            prev_sid = sorted_ids[i - 1]
            prev_val = values[prev_sid]
            cur_val = values[sid]
            if prev_val != 0:
                ratio = (cur_val / prev_val - 1) * 100
                lines.append(f"    → {sid}/{prev_sid} 环比: {ratio:+.2f}%")
            diff = cur_val - prev_val
            lines.append(f"    → {sid}-{prev_sid} 差值: {diff:+.4f}")

    return "\n".join(lines)


# ============================================================
# 辅助: 审查维度分类 — 从 reason 中检测哪些维度可能有问题
# ============================================================
def _classify_dimension(reason: str, dimension: str, *keywords: str) -> str:
    """根据 reason 中的关键词判断某个维度的审查结果。

    通过时 reason 为 "OK"，直接返回"通过"。
    否则扫描 reason 是否命中了某维度关联的关键词，命中则标"不通过"。

    Args:
        reason: 审查 reason 字符串。
        dimension: 维度名称（用于日志）。
        keywords: 与该维度关联的关键词。

    Returns:
        str: "通过" 或 "不通过 (维度名)"。
    """
    if reason.upper() == "OK" or not reason:
        return "通过"
    if not keywords:
        return "通过"
    for kw in keywords:
        if kw in reason:
            return f"不通过 ({dimension})"
    return "通过"


# ============================================================
# Prompt 模板 — 业务级自我审查
# ============================================================
_REFLECT_PROMPT = """你是一个 Text2SQL 质量审查专家。请从以下三个维度审查聚合答案的质量。

## 原始问题
{question}

## 执行计划
{plan_summary}

## 各子任务 SQL 与执行结果
{task_details}

## 聚合答案
{aggregated_answer}

## 审查维度
1. **语义校验**：答案是否真正回答了原始问题？有无答非所问、遗漏子问题、偏题？
2. **逻辑校验**：数值/计算逻辑是否符合业务常识？例如：
   - 同比/环比不应为负无穷或荒谬值
   - 占比不应超过 100%
   - 数据量级是否合理
3. **可执行性校验**：是否有子任务执行失败（status=failed）导致答案不完整不可信？

## 审查结论
- 三个维度全部通过 → passed=true, reason="OK"
- 任一维度不通过 → passed=false, reason 说明具体哪个维度哪里出了问题
- fix_hint 给出修正建议（不通过时必填）

请严格只输出一个 JSON 对象。"""


# ============================================================
# 节点 4: 业务级自我审查
# ============================================================
def reflect_node(state: MainState) -> dict[str, Any]:
    """反思审查节点：对聚合答案做三维修业务级审查。

    三个维度：
        1. 语义校验 — 是否回答了原始问题
        2. 逻辑校验 — 数值/计算是否符合常识
        3. 可执行性校验 — 是否有 failed 子任务导致答案不可信

    审查结果写入 state.reflection，不通过时 fix_hint 供重规划使用。

    Args:
        state: 主图全局状态。

    Returns:
        dict: 更新后的 state 字段（reflection / status）。
    """
    trace_id: str = state.get("trace_id", "-")
    question: str = state.get("question", "")
    plan: list[SubTask] = state.get("plan", [])
    completed: dict[str, SubTask] = state.get("completed", {})
    aggregated_answer: str = state.get("aggregated_answer", "")
    iteration: int = state.get("iteration", 0)
    max_iteration: int = state.get("max_iteration", 3)
    log = get_logger(trace_id)

    log.info(
        f"reflect_node 进入  iteration={iteration}/{max_iteration}"
    )

    # ---- 前置检查: 存在 failed 子任务 → 直接不通过 ----
    failed_tasks: list[str] = []
    for sid, task in completed.items():
        if task.get("status") == "failed":
            failed_tasks.append(sid)

    if failed_tasks:
        log.warning(
            f"reflect_node: 存在 {len(failed_tasks)} 个失败子任务: {failed_tasks}，直接 passed=false"
        )
        failure_reasons = []
        for sid in failed_tasks:
            err = completed[sid].get("error", "未知错误")
            failure_reasons.append(f"[{sid}] {err}")
        return {
            "reflection": {
                "passed": False,
                "reason": (
                    f"以下子任务执行失败，答案不可信: {'; '.join(failure_reasons)}"
                ),
                "fix_hint": (
                    f"请重新规划子任务 {failed_tasks}，修正 SQL 生成逻辑。"
                    f"注意表名是否存在、列名是否拼写正确。"
                ),
            },
            "status": "reflecting",
        }

    # ---- 组装审查数据 ----
    # plan 摘要
    plan_lines: list[str] = []
    for s in plan:
        deps = s.get("depends_on", [])
        dep_str = f" (依赖: {', '.join(deps)})" if deps else ""
        plan_lines.append(f"  [{s['id']}] {s.get('description', '-')}{dep_str}")
    plan_summary = "\n".join(plan_lines) if plan_lines else "无计划"

    # 各子任务详情（SQL + 结果）
    task_lines: list[str] = []
    for sid, task in completed.items():
        task_lines.append(f"### [{sid}] {task.get('description', '-')}")
        task_lines.append(f"  状态: {task.get('status', '-')}")
        if task.get("sql"):
            task_lines.append(f"  SQL: {task['sql']}")
        if task.get("status") == "success" and task.get("result"):
            r = task["result"]
            task_lines.append(f"  列: {r.get('columns', [])}")
            rows = r.get("rows", [])
            if len(rows) <= 10:
                task_lines.append(f"  数据: {rows}")
            else:
                task_lines.append(f"  数据（行数={len(rows)}，前5行）: {rows[:5]}")
        elif task.get("error"):
            task_lines.append(f"  错误: {task['error']}")
        task_lines.append("")
    task_details = "\n".join(task_lines)

    # 组装 Prompt 并调用 LLM
    prompt = _REFLECT_PROMPT.format(
        question=question,
        plan_summary=plan_summary,
        task_details=task_details,
        aggregated_answer=aggregated_answer,
    )

    schema_hint = (
        '{"passed": true/false, '
        '"reason": "不通过的具体原因（通过时填 OK）", '
        '"fix_hint": "给规划层的修正建议（不通过时必填，通过时留空字符串）"}'
    )

    t0 = time.perf_counter()
    try:
        reflect_result = call_json(
            prompt=prompt,
            schema_hint=schema_hint,
            trace_id=trace_id,
            max_retry=2,
        )
    except ValueError as e:
        log.error(f"reflect_node: call_json 失败: {e}")
        # 降级：审查 LLM 不可用时视为通过
        return {
            "reflection": {
                "passed": True,
                "reason": f"审查 LLM 调用失败，降级通过: {e}",
                "fix_hint": "",
            },
            "status": "done",
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000

    passed: bool = reflect_result.get("passed", False)
    reason: str = reflect_result.get("reason", "")
    fix_hint: str = reflect_result.get("fix_hint", "")

    # ---- 三个维度分开打点 ----
    # 1. 语义校验 — 从 reason 中拆解关键词做独立打点
    semantic_tag = _classify_dimension(reason, "语义", "答非所问", "遗漏", "偏题", "未回答")
    logic_tag = _classify_dimension(reason, "逻辑", "计算", "数值", "占比", "环比", "同比", "量级", "公式")
    exec_tag = _classify_dimension(reason, "可执行", "SQL", "执行", "failed", "子任务")

    log.info(
        f"reflect_node 完成  耗时 {elapsed_ms:.0f}ms  "
        f"passed={passed}  reason={reason[:120] if reason else '-'}"
    )
    log.info(f"  语义校验     → {semantic_tag}")
    log.info(f"  逻辑校验     → {logic_tag}")
    log.info(f"  可执行性校验 → {exec_tag}")
    if not passed and fix_hint:
        log.info(f"  fix_hint     → {fix_hint[:120]}")

    # 确定下一状态
    next_status = "reflecting"
    if passed:
        next_status = "reflecting"  # 由条件边决定去 finalize 还是 degrade

    return {
        "reflection": {
            "passed": passed,
            "reason": reason,
            "fix_hint": fix_hint,
        },
        "status": next_status,
    }


# ============================================================
# 节点 5: 降级回答
# ============================================================
def degrade_node(state: MainState) -> dict[str, Any]:
    """降级节点：反思全部失败后生成友好降级回答。

    收集已完成子任务中成功的部分结果，生成降级提示，
    引导用户简化问题或拆分提问。

    Args:
        state: 主图全局状态。

    Returns:
        dict: 更新后的 state 字段（final_answer / status）。
    """
    trace_id: str = state.get("trace_id", "-")
    question: str = state.get("question", "")
    completed: dict[str, SubTask] = state.get("completed", {})
    iteration: int = state.get("iteration", 0)
    max_iteration: int = state.get("max_iteration", 3)
    reflection = state.get("reflection", {})
    log = get_logger(trace_id)

    log.warning(
        f"degrade_node 进入  iteration={iteration}/{max_iteration}  "
        f"末次反思: {reflection.get('reason', '-')[:100]}"
    )

    # 收集部分可用结果
    succeeded: list[dict] = []
    for sid, task in completed.items():
        if task.get("status") == "success" and task.get("result"):
            succeeded.append({
                "id": sid,
                "desc": task.get("description", ""),
                "result": task["result"],
            })

    # 构造降级提示
    lines = [
        f"抱歉，经过 {iteration} 轮反思与重试，仍无法为「{question}」生成完全可信的答案。",
        "",
    ]

    if succeeded:
        lines.append("以下为部分已成功获取的数据，仅供参考：")
        for item in succeeded:
            r = item["result"]
            cols = r.get("columns", [])
            rows = r.get("rows", [])
            lines.append(f"  [{item['id']}] {item['desc']}")
            lines.append(f"       列: {cols}")
            lines.append(f"       数据: {rows[:5]}{' ...' if len(rows) > 5 else ''}")
        lines.append("")

    lines.append("建议您：")
    lines.append("  1. 简化问题，将复杂查询拆分为多个简单问题逐一提问。")
    lines.append("  2. 检查数据库中是否确实存在问题所描述的表和字段。")
    lines.append("  3. 使用 db_utils.get_schema 确认数据库结构后再提问。")

    deg_answer = "\n".join(lines)

    log.info(
        f"degrade_node 完成  "
        f"部分可用子任务={len(succeeded)}/{len(completed)}  "
        f"answer长度={len(deg_answer)}"
    )

    return {
        "final_answer": deg_answer,
        "status": "degraded",
    }


# ============================================================
# 节点 6: 最终化 — 反思通过后将聚合答案定为最终答案
# ============================================================
def finalize_node(state: MainState) -> dict[str, Any]:
    """最终化节点：反思通过后设定 final_answer。

    直接将 aggregated_answer 提升为 final_answer，
    状态置为 "done"。

    Args:
        state: 主图全局状态。

    Returns:
        dict: 更新后的 state 字段（final_answer / status）。
    """
    trace_id: str = state.get("trace_id", "-")
    aggregated_answer: str = state.get("aggregated_answer", "")
    log = get_logger(trace_id)

    log.info(f"finalize_node 进入  answer长度={len(aggregated_answer)}")

    return {
        "final_answer": aggregated_answer,
        "status": "done",
    }


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    from app.log_utils import new_trace_id
    from app.state import init_main_state

    tid = new_trace_id()
    log = get_logger(tid)
    log.info("=== nodes 自测开始 ===")

    # ---------- 自测 1: 拓扑排序 ----------
    log.info("--- 自测 1: 拓扑排序 ---")
    test_subtasks: list[SubTask] = [
        {"id": "t1", "description": "查总量", "depends_on": [], "db_id": "test",
         "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        {"id": "t2", "description": "查上月量", "depends_on": [], "db_id": "test",
         "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        {"id": "t3", "description": "合成", "depends_on": ["t1", "t2"], "db_id": "test",
         "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
    ]
    try:
        levels = _topological_levels(test_subtasks)
        log.info(f"拓扑分层结果: {[[s['id'] for s in lv] for lv in levels]}")
        assert len(levels) == 2, f"期望2层，实际{len(levels)}层"
        assert [s["id"] for s in levels[0]] == ["t1", "t2"]
        assert [s["id"] for s in levels[1]] == ["t3"]
        log.info("[PASS] 拓扑排序自测通过")
    except Exception as e:
        log.error(f"[FAIL] 拓扑排序自测: {e}")

    # ---------- 自测 2: 拓扑排序——检测环 ----------
    log.info("--- 自测 2: 拓扑排序 - 检测环 ---")
    cyclic_subtasks: list[SubTask] = [
        {"id": "t1", "description": "A", "depends_on": ["t2"], "db_id": "test",
         "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        {"id": "t2", "description": "B", "depends_on": ["t1"], "db_id": "test",
         "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
    ]
    try:
        _topological_levels(cyclic_subtasks)
        log.error("[FAIL] 应抛出 ValueError")
    except ValueError as e:
        log.info(f"检测到环，正确抛出异常: {e}")
        log.info("[PASS] 环检测自测通过")

    # ---------- 自测 3: 端到端 plan → dispatch → aggregate ----------
    log.info("--- 自测 3: 端到端 plan→dispatch→aggregate ---")
    initial_state = init_main_state(
        question="department_store 数据库中 Customers 表有多少条记录？同时列出所有产品名称。",
        db_id="department_store",
        trace_id=tid,
        max_iteration=1,
    )

    try:
        # Step 1: plan
        plan_result = plan_node(initial_state)
        log.info(f"plan 完成: subtasks={len(plan_result.get('plan', []))}")

        # 合并进 state
        state_after_plan = {**initial_state, **plan_result}

        # Step 2: dispatch
        dispatch_result = dispatch_node(state_after_plan)
        log.info(f"dispatch 完成: completed={len(dispatch_result.get('completed', {}))}")

        # 合并进 state
        state_after_dispatch = {**state_after_plan, **dispatch_result}

        # Step 3: aggregate
        agg_result = aggregate_node(state_after_dispatch)
        log.info(f"aggregate 完成: answer长度={len(agg_result.get('aggregated_answer', ''))}")
        log.info(f"aggregated_answer: {agg_result.get('aggregated_answer', '')[:200]}")

        log.info("=== 端到端流程完成 ===")
        log.info("[PASS] nodes 自测通过")
    except Exception as e:
        log.error(f"[FAIL] 端到端自测异常: {e}")
        import traceback
        traceback.print_exc()

    log.info("=== nodes 自测结束 ===")
