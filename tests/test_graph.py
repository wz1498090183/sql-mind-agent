"""
测试 app.graph — 反思路由逻辑。
"""

import pytest

from app.graph import build_main_graph, route_after_reflect
from app.state import MainState, init_main_state


# ------------------------------------------------------------
# route_after_reflect 测试（纯函数，不需要 LLM）
# ------------------------------------------------------------

class TestRouteAfterReflect:
    """reflect 节点后三路条件边路由逻辑测试。"""

    def test_passed_true_routes_to_finalize(self) -> None:
        """反思通过 → finalize。"""
        state: MainState = init_main_state("Q", "db", "tid")
        state["reflection"] = {"passed": True, "reason": "OK", "fix_hint": ""}
        assert route_after_reflect(state) == "finalize"

    def test_passed_false_with_retries_left_routes_to_retry(self) -> None:
        """反思不通过 + 迭代未达上限 → retry。"""
        state: MainState = init_main_state("Q", "db", "tid")
        state["reflection"] = {"passed": False, "reason": "结果不对", "fix_hint": "改一下"}
        state["iteration"] = 0
        state["max_iteration"] = 3
        assert route_after_reflect(state) == "retry"

    def test_passed_false_at_limit_routes_to_degrade(self) -> None:
        """反思不通过 + 迭代已达上限 → degrade。"""
        state: MainState = init_main_state("Q", "db", "tid")
        state["reflection"] = {"passed": False, "reason": "反复失败", "fix_hint": ""}
        state["iteration"] = 2  # max_iteration=3, iteration=2 → 已达上限
        state["max_iteration"] = 3
        assert route_after_reflect(state) == "degrade"


class TestBuildMainGraph:
    """主图构建测试。"""

    def test_graph_compiles(self) -> None:
        """build_main_graph 返回编译成功的图。"""
        graph = build_main_graph()
        assert graph is not None
        # LangGraph compiled graph 应该有 invoke 方法
        assert hasattr(graph, "invoke")

    def test_graph_has_all_nodes(self) -> None:
        """编译后的图包含全部 6 个节点。"""
        graph = build_main_graph()
        # 通过节点名称在图中存在的角度验证
        node_names = graph.get_graph().nodes if hasattr(graph, "get_graph") else None
        if node_names is not None:
            expected = {"plan", "dispatch", "aggregate", "reflect", "finalize", "degrade"}
            actual = set(node_names.keys())
            assert actual >= expected, f"缺少节点: {expected - actual}"
