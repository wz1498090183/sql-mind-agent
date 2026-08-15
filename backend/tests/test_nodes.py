"""
测试 app.nodes — 主图节点核心逻辑。
覆盖: 拓扑排序（正常/环检测），plan_node / aggregate_node 功能。
"""
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.nodes import (
    _topological_levels,
    _pre_calc_summary,
    _classify_dimension,
    plan_node,
    aggregate_node,
)
from app.state import SubTask, init_main_state


TEST_DB_ID = "department_store"

# ============================================================
# 拓扑排序测试（不需要 LLM/SQL，纯函数）
# ============================================================

class TestTopologicalLevels:
    """拓扑排序和环检测测试。"""

    def test_linear_dependency(self) -> None:
        """线性依赖: t1 → t2 → t3 — 应为 3 层。"""
        tasks: list[SubTask] = [
            {"id": "t1", "description": "A", "depends_on": [], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t2", "description": "B", "depends_on": ["t1"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t3", "description": "C", "depends_on": ["t2"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        ]
        levels = _topological_levels(tasks)
        assert len(levels) == 3
        assert [s["id"] for s in levels[0]] == ["t1"]
        assert [s["id"] for s in levels[1]] == ["t2"]
        assert [s["id"] for s in levels[2]] == ["t3"]

    def test_parallel_independent(self) -> None:
        """无依赖的两个子任务 — 应在同一层并行。"""
        tasks: list[SubTask] = [
            {"id": "t1", "description": "A", "depends_on": [], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t2", "description": "B", "depends_on": [], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        ]
        levels = _topological_levels(tasks)
        assert len(levels) == 1
        layer0_ids = {s["id"] for s in levels[0]}
        assert layer0_ids == {"t1", "t2"}

    def test_diamond_dependency(self) -> None:
        """菱形依赖: t1,t2 并行 → t3 依赖二者 → t4 依赖 t3。"""
        tasks: list[SubTask] = [
            {"id": "t1", "description": "左", "depends_on": [], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t2", "description": "右", "depends_on": [], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t3", "description": "汇合", "depends_on": ["t1", "t2"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t4", "description": "最终", "depends_on": ["t3"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        ]
        levels = _topological_levels(tasks)
        assert len(levels) == 3
        assert {s["id"] for s in levels[0]} == {"t1", "t2"}
        assert [s["id"] for s in levels[1]] == ["t3"]
        assert [s["id"] for s in levels[2]] == ["t4"]

    def test_cycle_detection(self) -> None:
        """环检测 — A→B→A 循环依赖应抛出 ValueError。"""
        tasks: list[SubTask] = [
            {"id": "t1", "description": "A", "depends_on": ["t2"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
            {"id": "t2", "description": "B", "depends_on": ["t1"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        ]
        with pytest.raises(ValueError, match="循环依赖|依赖环"):
            _topological_levels(tasks)

    def test_missing_dependency(self) -> None:
        """依赖不存在的子任务 — 应抛出 ValueError。"""
        tasks: list[SubTask] = [
            {"id": "t1", "description": "A", "depends_on": ["t99"], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        ]
        with pytest.raises(ValueError, match="不在计划中|depends"):
            _topological_levels(tasks)

    def test_single_task(self) -> None:
        """单个子任务 — 一层即可。"""
        tasks: list[SubTask] = [
            {"id": "t1", "description": "Solo", "depends_on": [], "db_id": "test",
             "sql": None, "result": None, "status": "pending", "error": None, "retry_count": 0},
        ]
        levels = _topological_levels(tasks)
        assert len(levels) == 1
        assert levels[0][0]["id"] == "t1"


# ============================================================
# _pre_calc_summary 测试
# ============================================================

class TestPreCalcSummary:
    """数值预计算摘要函数测试。"""

    def test_two_values_diff(self) -> None:
        """两个单行单列数值 — 应输出差值和比率。"""
        completed: dict[str, SubTask] = {
            "t1": {
                "id": "t1", "description": "当前", "depends_on": [],
                "db_id": "test", "sql": None, "error": None, "retry_count": 0,
                "status": "success",
                "result": {"success": True, "columns": ["cnt"], "rows": [(200,)]},
            },
            "t2": {
                "id": "t2", "description": "上期", "depends_on": [],
                "db_id": "test", "sql": None, "error": None, "retry_count": 0,
                "status": "success",
                "result": {"success": True, "columns": ["cnt"], "rows": [(100,)]},
            },
        }
        summary = _pre_calc_summary(completed)
        assert "t1: 200.0" in summary
        assert "t2: 100.0" in summary
        assert "环比" in summary or "+100" in summary

    def test_single_value_no_summary(self) -> None:
        """只有一条数据 — 不成对比，返回空字符串。"""
        completed: dict[str, SubTask] = {
            "t1": {
                "id": "t1", "description": "仅一条", "depends_on": [],
                "db_id": "test", "sql": None, "error": None, "retry_count": 0,
                "status": "success",
                "result": {"success": True, "columns": ["cnt"], "rows": [(42,)]},
            },
        }
        assert _pre_calc_summary(completed) == ""


# ============================================================
# _classify_dimension 测试
# ============================================================

class TestClassifyDimension:
    """审查维度分类函数测试。"""

    def test_ok_reason_passes(self) -> None:
        """reason 为 OK → 三个维度全部通过。"""
        assert _classify_dimension("OK", "语义", "答非所问") == "通过"
        assert _classify_dimension("OK", "逻辑") == "通过"

    def test_semantic_failure(self) -> None:
        """reason 含'答非所问' → 语义维度不通过。"""
        result = _classify_dimension("答案答非所问，没有回答用户的问题", "语义", "答非所问", "遗漏", "偏题")
        assert "不通过" in result
        assert "语义" in result

    def test_logic_failure(self) -> None:
        """reason 含'计算' → 逻辑维度不通过。"""
        result = _classify_dimension("同比环比计算结果错误", "逻辑", "计算", "数值", "占比")
        assert "不通过" in result

    def test_empty_reason_passes(self) -> None:
        """空 reason → 通过。"""
        assert _classify_dimension("", "逻辑", "计算") == "通过"
