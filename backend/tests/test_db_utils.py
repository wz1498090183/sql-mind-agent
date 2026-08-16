"""
测试 app.db_utils — 数据库工具模块。
覆盖: execute_sql 正常/异常/只读校验，get_schema / get_schema_dict。
"""

import pytest

from app.db_utils import execute_sql, get_schema, get_schema_dict, get_value_samples

# 测试用数据库 ID（需在 SPIDER_DB_ROOT 下存在）
TEST_DB_ID = "department_store"


class TestExecuteSql:
    """execute_sql 功能测试。"""

    def test_simple_select(self) -> None:
        """简单 SELECT 查询 — 成功返回结构化结果。"""
        result = execute_sql(TEST_DB_ID, "SELECT 1 AS num")
        assert result["success"] is True
        assert result["columns"] == ["num"]
        assert result["rows"] == [(1,)]
        assert result["error"] is None

    def test_select_from_real_table(self) -> None:
        """从真实表查询 — 返回列名和数据。"""
        # 先获取 Schema 确认有表存在
        schema_dict = get_schema_dict(TEST_DB_ID)
        if not schema_dict:
            pytest.skip(f"{TEST_DB_ID} 中无用户表，跳过")

        first_table = next(iter(schema_dict))
        result = execute_sql(TEST_DB_ID, f"SELECT * FROM {first_table} LIMIT 2")
        assert result["success"] is True
        assert len(result["columns"]) > 0

    def test_sql_syntax_error(self) -> None:
        """SQL 语法错误 — 返回 success=False 且含错误信息。"""
        result = execute_sql(TEST_DB_ID, "SELEC * FROM nonexistent")
        assert result["success"] is False
        assert result["error"] is not None

    def test_nonexistent_table(self) -> None:
        """查询不存在的表 — 返回成功失败和明确的错误信息。"""
        result = execute_sql(TEST_DB_ID, "SELECT * FROM nonexistent_table_xyz")
        assert result["success"] is False
        assert result["error"] is not None

    def test_reject_insert(self) -> None:
        """拦截 INSERT 语句 — 即使表存在也拒绝。"""
        result = execute_sql(TEST_DB_ID, "INSERT INTO foo VALUES (1)")
        assert result["success"] is False
        assert "禁止" in result.get("error", "")

    def test_reject_delete(self) -> None:
        """拦截 DELETE 语句。"""
        result = execute_sql(TEST_DB_ID, "DELETE FROM Customers WHERE 1=1")
        assert result["success"] is False
        assert "禁止" in result.get("error", "")

    def test_reject_update(self) -> None:
        """拦截 UPDATE 语句。"""
        result = execute_sql(TEST_DB_ID, "UPDATE Customers SET name='x'")
        assert result["success"] is False
        assert "禁止" in result.get("error", "")

    def test_allow_with_clause(self) -> None:
        """允许 WITH 子句（CTE）。"""
        result = execute_sql(
            TEST_DB_ID,
            "WITH t AS (SELECT 1 AS n) SELECT * FROM t",
        )
        assert result["success"] is True

    def test_reject_drop(self) -> None:
        """拦截 DROP 语句。"""
        result = execute_sql(TEST_DB_ID, "DROP TABLE Customers")
        assert result["success"] is False
        assert "禁止" in result.get("error", "")


class TestGetSchema:
    """Schema 提取功能测试。"""

    def test_returns_string(self) -> None:
        """get_schema 返回非空字符串。"""
        schema = get_schema(TEST_DB_ID)
        assert isinstance(schema, str)
        assert len(schema) > 0
        assert "CREATE TABLE" in schema

    def test_missing_db_raises(self) -> None:
        """不存在的 db_id 抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            get_schema("nonexistent_db_xyz_123")


class TestGetSchemaDict:
    """Schema Dict 功能测试。"""

    def test_returns_dict(self) -> None:
        """get_schema_dict 返回 {表名: [列名, ...]} 映射。"""
        schema_dict = get_schema_dict(TEST_DB_ID)
        assert isinstance(schema_dict, dict)
        assert len(schema_dict) > 0
        for table_name, columns in schema_dict.items():
            assert isinstance(table_name, str)
            assert isinstance(columns, list)
            assert len(columns) > 0


class TestGetValueSamples:
    """值检索 get_value_samples 功能测试。"""

    def test_returns_non_empty(self) -> None:
        """department_store 含类别列，值检索应返回非空样本。"""
        samples = get_value_samples(TEST_DB_ID)
        assert isinstance(samples, str)
        assert len(samples) > 0

    def test_contains_category_column(self) -> None:
        """样本应覆盖 product_type_code 这类低基数类别列。"""
        samples = get_value_samples(TEST_DB_ID)
        assert "product_type_code" in samples

    def test_respects_max_cols(self) -> None:
        """max_cols=1 时最多返回一列的样本。"""
        samples = get_value_samples(TEST_DB_ID, max_cols=1)
        # 每列样本占一行 "- 表.列: ..."，max_cols=1 时仅 1 行
        lines = [ln for ln in samples.splitlines() if ln.startswith("- ")]
        assert len(lines) == 1
