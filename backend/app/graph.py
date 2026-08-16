"""
LangGraph 主图编排（Day2+ 版本）。
链路: START → plan → dispatch → aggregate → reflect
       → [passed → finalize → END | 未达上限 → retry(iteration+1) → plan | 达上限 → degrade → END]
"""

import sys
from pathlib import Path
from typing import Literal

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langgraph.graph import END, START, StateGraph

from app.nodes import (
    aggregate_node,
    degrade_node,
    dispatch_node,
    finalize_node,
    plan_node,
    reflect_node,
)
from app.state import MainState, Reflection


def route_after_reflect(state: MainState) -> Literal["finalize", "retry", "degrade"]:
    """reflect 节点后的条件边函数。

    决策逻辑：
        - passed == True  → "finalize"（答案可信，直接 END）
        - passed == False 且 iteration < max_iteration - 1
          → "retry"（iteration+1，带上 fix_hint 回 plan 重规划）
        - passed == False 且 iteration >= max_iteration - 1
          → "degrade"（已达上限，降级回答）

    Args:
        state: 主图全局状态。

    Returns:
        路由目标字符串。
    """
    reflection: Reflection | None = state.get("reflection")
    iteration: int = state.get("iteration", 0)
    max_iteration: int = state.get("max_iteration", 3)

    if reflection and reflection.get("passed", False):
        return "finalize"

    if iteration < max_iteration - 1:
        return "retry"

    return "degrade"


def _on_retry(state: MainState) -> dict:
    """retry 分支的预处理：iteration+1，清空上一轮 plan/completed/aggregated_answer，
    但保留 reflection.fix_hint 供 plan_node 参考。
    """
    return {
        "iteration": state.get("iteration", 0) + 1,
        "plan": [],
        "completed": {},
        "aggregated_answer": None,
    }


def build_main_graph():
    """构建完整主图: START → plan → dispatch → aggregate → reflect → 三路条件边。

    三路分支（route_after_reflect）:
        - finalize: reflect 通过 → finalize_node → END
        - retry:    未达上限 → iteration+1，清中间态，回到 plan_node
        - degrade:  已达上限 → degrade_node → END

    plan_node 重规划时会读取 state.reflection 的 reason/fix_hint。

    Returns:
        CompiledGraph: 已编译的 LangGraph 执行图。
    """
    graph = StateGraph(MainState)

    # 添加全部 6 个节点 + retry 预处理节点
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("degrade", degrade_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("retry", _on_retry)

    # 主链路: START → plan → dispatch → aggregate → reflect
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "dispatch")
    graph.add_edge("dispatch", "aggregate")
    graph.add_edge("aggregate", "reflect")

    # reflect 后三路条件边
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "finalize": "finalize",
            "retry": "retry",
            "degrade": "degrade",
        },
    )

    # retry 预处理（iteration+1、清中间态）后回到 plan 重新规划
    graph.add_edge("retry", "plan")

    # finalize / degrade 都通向 END
    graph.add_edge("finalize", END)
    graph.add_edge("degrade", END)

    return graph.compile()
