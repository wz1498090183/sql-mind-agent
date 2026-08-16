"""
测试 app.sql_subgraph — SQL 求解子图模块。
覆盖: SQL 清洗、高危关键字拦截、校验逻辑。
"""


from app.sql_subgraph import (
    _clean_sql,
    _contains_dangerous_sql,
    _references_undefined_upstream,
)


class TestCleanSql:
    """SQL 清洗函数测试。"""

    def test_strip_whitespace(self) -> None:
        """去除首尾空白。"""
        assert _clean_sql("  SELECT 1  ") == "SELECT 1"

    def test_remove_markdown_sql_block(self) -> None:
        """去掉 ```sql ... ``` 代码块包裹。"""
        raw = "```sql\nSELECT * FROM t;\n```"
        assert _clean_sql(raw) == "SELECT * FROM t;"

    def test_remove_markdown_plain_block(self) -> None:
        """去掉 ``` ... ``` 代码块包裹（无语言标注）。"""
        raw = "```\nSELECT 1\n```"
        assert _clean_sql(raw) == "SELECT 1"

    def test_keep_bare_sql(self) -> None:
        """纯 SQL 直接返回。"""
        sql = "SELECT COUNT(*) FROM Customers;"
        assert _clean_sql(sql) == sql

    def test_only_remove_fenced_sql(self) -> None:
        """保留 Markdown 三反引号之外的 SQL 内容。"""
        raw = "这是说明\n```sql\nSELECT 1\n```\n另一段说明"
        cleaned = _clean_sql(raw)
        # 只取代码块内内容
        assert "SELECT 1" in cleaned
        assert "这是说明" not in cleaned


class TestContainsDangerousSql:
    """高危关键字检测测试。"""

    def test_drop_is_dangerous(self) -> None:
        """DROP 关键字标记为危险。"""
        assert _contains_dangerous_sql("DROP TABLE Customers") is not None

    def test_delete_is_dangerous(self) -> None:
        """DELETE 关键字标记为危险。"""
        assert _contains_dangerous_sql("DELETE FROM Customers WHERE id=1") is not None

    def test_update_is_dangerous(self) -> None:
        """UPDATE 关键字标记为危险。"""
        assert _contains_dangerous_sql("UPDATE Customers SET name='x'") is not None

    def test_insert_is_dangerous(self) -> None:
        """INSERT 关键字标记为危险。"""
        assert _contains_dangerous_sql("INSERT INTO t VALUES (1)") is not None

    def test_select_is_safe(self) -> None:
        """SELECT 语句安全。"""
        assert _contains_dangerous_sql("SELECT * FROM Customers") is None

    def test_cte_is_safe(self) -> None:
        """WITH 子句（CTE）安全。"""
        assert _contains_dangerous_sql(
            "WITH t AS (SELECT * FROM Customers) SELECT * FROM t"
        ) is None

    def test_keyword_in_column_name_is_safe(self) -> None:
        """表名/列名中含 INSERT/DELETE 等子串不应误判。
        例如列名 'is_deleted' 不应匹配 DELETE 关键字。
        """
        assert _contains_dangerous_sql(
            "SELECT id, is_deleted, updated_at FROM Customers WHERE is_deleted = 1"
        ) is None

    def test_case_insensitive(self) -> None:
        """大小写不敏感检测。"""
        assert _contains_dangerous_sql("drop table x") is not None
        assert _contains_dangerous_sql("Delete from X") is not None


class TestReferencesUndefinedUpstream:
    """上游子任务 id 误当表名引用的检测测试。"""

    _UPSTREAM = {
        "t1": {"success": True, "columns": ["customer_id", "order_count"], "rows": [(1, 3)]}
    }

    def test_undefined_upstream_id_detected(self) -> None:
        """FROM 直接引用上游 id（无 CTE 定义）→ 应检测到误用。"""
        sql = (
            "SELECT a.address_details "
            "FROM t1 "
            "JOIN Customer_Addresses ca ON t1.customer_id = ca.customer_id "
            "JOIN Addresses a ON ca.address_id = a.address_id"
        )
        assert _references_undefined_upstream(sql, self._UPSTREAM) == ["t1"]

    def test_cte_defined_id_is_not_misused(self) -> None:
        """上游 id 通过 WITH ... AS (...) 定义成 CTE 后再引用 → 不算误用。"""
        sql = (
            "WITH t1 AS ("
            "  SELECT customer_id, COUNT(order_id) AS order_count "
            "  FROM Customer_Orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 1"
            ") "
            "SELECT a.address_details FROM t1 "
            "JOIN Customer_Addresses ca ON t1.customer_id = ca.customer_id "
            "JOIN Addresses a ON ca.address_id = a.address_id"
        )
        assert _references_undefined_upstream(sql, self._UPSTREAM) == []

    def test_no_upstream_returns_empty(self) -> None:
        """无上游依赖 → 恒返回空列表。"""
        sql = "SELECT * FROM t1"
        assert _references_undefined_upstream(sql, None) == []
        assert _references_undefined_upstream(sql, {}) == []

    def test_case_insensitive(self) -> None:
        """上游 id 大小写不敏感匹配。"""
        sql = "SELECT * FROM T1 JOIN Addresses a ON 1=1"
        assert _references_undefined_upstream(sql, self._UPSTREAM) == ["T1"]
