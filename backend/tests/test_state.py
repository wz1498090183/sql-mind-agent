"""
测试 app.state — 状态定义与工厂函数。
覆盖: SubTask / MainState 字段完整性，init_main_state 工厂函数。
"""


from app.state import (
    SubTask,
    init_main_state,
)


class TestSubTask:
    """SubTask 结构的基础单元测试。"""

    def test_create_minimal_subtask(self) -> None:
        """最简 SubTask 创建 — 仅必填字段。"""
        task = SubTask(
            id="t1",
            description="查询客户总数",
            depends_on=[],
            db_id="test_db",
            sql=None,
            result=None,
            status="pending",
            error=None,
            retry_count=0,
        )
        assert task["id"] == "t1"
        assert task["description"] == "查询客户总数"
        assert task["depends_on"] == []
        assert task["db_id"] == "test_db"
        assert task["sql"] is None
        assert task["result"] is None
        assert task["status"] == "pending"
        assert task["error"] is None
        assert task["retry_count"] == 0

    def test_subtask_with_dependencies(self) -> None:
        """带依赖的 SubTask — depends_on 包含多个上游任务。"""
        task = SubTask(
            id="t3",
            description="聚合 t1 和 t2 结果",
            depends_on=["t1", "t2"],
            db_id="test_db",
            sql=None,
            result=None,
            status="pending",
            error=None,
            retry_count=0,
        )
        assert "t1" in task["depends_on"]
        assert "t2" in task["depends_on"]
        assert len(task["depends_on"]) == 2


class TestInitMainState:
    """init_main_state 工厂函数测试。"""

    def test_basic_initialization(self) -> None:
        """基本初始化 — 各字段具备合理初始值。"""
        state = init_main_state(
            question="测试问题",
            db_id="test_db",
            trace_id="abc12345",
        )
        assert state["trace_id"] == "abc12345"
        assert state["question"] == "测试问题"
        assert state["db_id"] == "test_db"
        assert state["plan"] == []
        assert state["completed"] == {}
        assert state["aggregated_answer"] is None
        assert state["reflection"] is None
        assert state["iteration"] == 0
        assert state["max_iteration"] == 3  # 默认值
        assert state["final_answer"] is None
        assert state["status"] == "planning"
        assert state["_completed_tasks"] == []

    def test_custom_max_iteration(self) -> None:
        """自定义最大迭代轮次。"""
        state = init_main_state(
            question="测试",
            db_id="test_db",
            trace_id="abc",
            max_iteration=5,
        )
        assert state["max_iteration"] == 5

    def test_returns_typed_dict(self) -> None:
        """返回值符合 MainState TypedDict。"""
        state = init_main_state("Q", "db", "tid")
        # 验证所有必需的 key 都存在
        expected_keys = {
            "trace_id", "question", "db_id",
            "plan", "completed", "aggregated_answer",
            "reflection", "iteration", "max_iteration",
            "final_answer", "status", "_completed_tasks",
        }
        assert set(state.keys()) >= expected_keys
