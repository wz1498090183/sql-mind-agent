"""
LangGraph 多步骤 Text2SQL Agent 的状态定义模块。

本模块定义了 TypedDict 类型和工厂函数，覆盖：
- SubTask：单个子任务的完整生命周期数据
- SqlResult：SQL 执行结果结构
- Reflection：反思审查结果结构
- SubGraphState：SQL 求解子图（生成→校验→执行→重试）的内部状态
- MainState：主图（拆解→调度→聚合→反思）的全局状态

生命周期中每个字段由哪个节点写入，见字段上方注释。
"""

import operator
from typing import Annotated, TypedDict


# ============================================================
# 0. 基础数据结构
# ============================================================
class SqlResult(TypedDict):
    """SQL 执行结果结构（db_utils.execute_sql 返回值）。"""

    success: bool
    columns: list[str]
    rows: list[tuple]
    error: str | None


class Reflection(TypedDict):
    """反思审查结果结构（由 reflect_node 写入，route_after_reflect 消费）。"""

    passed: bool
    reason: str
    fix_hint: str

# ============================================================
# 1. SubTask — 单个子任务
# ============================================================
class SubTask(TypedDict):
    """计划中的一个子任务，由主图 planner 节点创建，其余字段逐步填充。"""

    # 写入节点: planner（主图） — 创建时写入
    id: str
    # 写入节点: planner（主图） — 创建时写入
    description: str
    # 写入节点: planner（主图） — 依赖分析后写入，空列表表示无依赖、可并行
    depends_on: list[str]
    # 写入节点: planner（主图） — 创建时写入
    db_id: str

    # 写入节点: gen_sql（子图） — SQL 生成后写入
    sql: str | None
    # 写入节点: execute_sql（子图 / db_utils） — 执行后写入
    result: SqlResult | None
    # 写入节点: planner（主图）初始为 "pending"；后续由 dispatch/supervisor 节点更新
    status: str
    # 写入节点: validate / execute（子图） — 校验或执行失败时写入
    error: str | None
    # 写入节点: execute（子图） — 每次重试时自增
    retry_count: int


# ============================================================
# 2. SubGraphState — SQL 求解子图内部状态
# ============================================================
class SubGraphState(TypedDict):
    """子图内部状态，承载单个 SubTask 的 SQL 生成→校验→执行→重试 闭环。"""

    # 写入节点: dispatch（主图） — 调度时从 plan 中取出并传入
    subtask: SubTask
    # 写入节点: get_schema 工具（子图） — 进入子图时拉取目标库的 CREATE TABLE DDL
    schema: str
    # 写入节点: dispatch（主图） — 从 subtask.db_id 复制
    db_id: str
    # 写入节点: dispatch（主图） — 收集 completed 中所有依赖子任务的 result 字段
    upstream_results: dict
    # 写入节点: execute_node（子图） — 成功时写入当前子任务，LangGraph 自动累加到主图
    #          必须与 MainState 同名 + 同注解（operator.add），Send 返回时自动合并
    _completed_tasks: Annotated[list[SubTask], operator.add]


# ============================================================
# 3. MainState — 主图全局状态
# ============================================================
class MainState(TypedDict):
    """主图全局状态，贯穿 计划→调度→执行→聚合→反思→重试 全流程。"""

    # 写入节点: 外部入口 / init_main_state() — 标识一次端到端查询链路
    trace_id: str
    # 写入节点: 外部入口 / init_main_state() — 用户输入
    question: str
    # 写入节点: 外部入口 / init_main_state() — 用户指定或路由推断
    db_id: str

    # 写入节点: planner（主图） — 拆解 complex question 为子任务列表
    plan: list[SubTask]
    # 写入节点: aggregator（主图） — 从 _completed_tasks 重建 {id: SubTask} 映射
    completed: dict[str, SubTask]

    # 写入节点: aggregator（主图） — 合并所有已完成子任务的结果
    aggregated_answer: str | None
    # 写入节点: reflector（主图） — 反思聚合答案是否通过校验
    reflection: Reflection | None

    # 写入节点: reflector（主图） — 每次反思时自增
    iteration: int
    # 写入节点: 外部入口 / init_main_state() — 可在初始化时覆盖
    max_iteration: int

    # 写入节点: reflector（主图） — 反思通过后写入最终答案
    final_answer: str | None
    # 写入节点: 各节点均可更新；planner 写 "planning"，dispatch 写 "executing"，
    #          reflector 写 "reflecting"/"done"/"degraded"
    status: str

    # 写入节点: 子图 execute_node 成功时累加写入
    #          Annotated[list, operator.add] 使 LangGraph 自动合并并行子图的返回值
    _completed_tasks: Annotated[list[SubTask], operator.add]


# ============================================================
# 4. 工厂函数
# ============================================================
def init_main_state(
    question: str,
    db_id: str,
    trace_id: str,
    max_iteration: int = 3,
) -> MainState:
    """创建初始 MainState，供主图入口调用。

    初始化后状态为 "planning"，plan/completed 为空，iteration 为 0，
    aggregated_answer / reflection / final_answer 均为 None。

    Args:
        question: 用户原始复杂业务问题。
        db_id: 目标数据库标识符。
        trace_id: 本次查询的唯一链路 ID。
        max_iteration: 最大反思迭代轮次，默认 3。

    Returns:
        MainState: 填充好初始值的全局状态字典。
    """
    return MainState(
        trace_id=trace_id,
        question=question,
        db_id=db_id,
        plan=[],
        completed={},
        aggregated_answer=None,
        reflection=None,
        iteration=0,
        max_iteration=max_iteration,
        final_answer=None,
        status="planning",
        _completed_tasks=[],
    )
