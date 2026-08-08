"""
SQLite 数据库操作工具模块。
为 Text2SQL Agent 提供数据库连接、只读查询和 Schema 提取功能，
全部基于 Python 标准库 sqlite3，不引入 ORM。
"""

import os
import sqlite3
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env 文件中的环境变量
_env_path = _project_root / ".env"
if _env_path.is_file():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())


def get_connection(db_id: str) -> sqlite3.Connection:
    """根据 db_id 创建 SQLite 数据库连接。

    从环境变量 SPIDER_DB_ROOT 拼接数据库文件路径：
        {SPIDER_DB_ROOT}/{db_id}/{db_id}.sqlite
    连接失败时抛出 FileNotFoundError 并附带完整缺失路径。

    Args:
        db_id: 数据库标识符，对应 Spider 数据集中的数据库名（如 "concert_singer"）。

    Returns:
        sqlite3.Connection: 已建立的数据库连接对象，使用 Row 工厂以便按列名访问。

    Raises:
        FileNotFoundError: 数据库文件不存在或路径不可访问。
    """
    root = os.environ.get("SPIDER_DB_ROOT", "./spider/database")
    # 相对路径基于项目根目录解析
    db_path = (Path(root) if Path(root).is_absolute() else _project_root / root).resolve()
    db_path = db_path / db_id / f"{db_id}.sqlite"

    if not db_path.is_file():
        raise FileNotFoundError(
            f"数据库文件不存在: {db_path.resolve()}\n"
            f"请确认 SPIDER_DB_ROOT（{root}）下存在目录 {db_id}/ 及文件 {db_id}.sqlite"
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def execute_sql(db_id: str, sql: str, timeout: float = 5.0) -> dict:
    """执行只读 SQL 查询，返回结构化结果字典。

    仅允许 SELECT 和 WITH 开头的只读语句（含 WITH RECURSIVE），
    其他 SQL 类型（INSERT/UPDATE/DELETE/DROP 等）直接拒绝。
    所有 sqlite3.Error 被捕获并转换为结构化 error 字段，不向上抛出异常。

    Args:
        db_id: 数据库标识符。
        sql: 待执行的 SQL 语句（仅允许 SELECT / WITH）。
        timeout: 数据库忙等待超时秒数，默认 5.0 秒。

    Returns:
        dict: 包含以下键的结构化结果：
            - success (bool): 是否执行成功。
            - columns (list[str]): 结果列名列表，失败时为空列表。
            - rows (list[tuple]): 查询结果行列表（每行为 tuple），失败时为空列表。
            - error (str | None): 错误信息，成功时为 None。
    """
    stripped = sql.strip()
    # 取足够长度覆盖 "WITH RECURSIVE" 前缀
    upper_prefix = stripped[:16].upper()
    if not (upper_prefix.startswith("SELECT") or upper_prefix.startswith("WITH")):
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "error": "仅允许只读查询（SELECT / WITH），禁止写操作",
        }

    try:
        conn = get_connection(db_id)
    except FileNotFoundError as e:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "error": str(e),
        }

    try:
        with conn:
            conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
            cursor = conn.execute(stripped)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            # sqlite3.Row 转 tuple 确保结果可序列化
            if rows and isinstance(rows[0], sqlite3.Row):
                rows = [tuple(row) for row in rows]
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "error": None,
            }
    except sqlite3.Error as e:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "error": f"SQL 执行错误: {e}",
        }
    finally:
        conn.close()


def get_schema(db_id: str) -> str:
    """获取数据库中所有用户表的 CREATE TABLE 语句。

    从 sqlite_master 系统表读取 type='table' 且非系统表的 sql 字段，
    拼接为一个字符串，用于注入 LLM Prompt 提供表结构上下文。

    Args:
        db_id: 数据库标识符。

    Returns:
        str: 所有 CREATE TABLE 语句，表之间用一个空行分隔。
    """
    conn = get_connection(db_id)
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
        schemas = [row["sql"] for row in rows]
        return "\n".join(schemas)
    finally:
        conn.close()


def get_schema_dict(db_id: str) -> dict[str, list[str]]:
    """获取数据库表结构字典，供程序化校验使用。

    通过 PRAGMA table_info 逐表获取列名，返回 {表名: [列名, ...]} 映射。

    Args:
        db_id: 数据库标识符。

    Returns:
        dict[str, list[str]]: 表名到列名列表的映射，键按表名排序。
    """
    conn = get_connection(db_id)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()

        result: dict[str, list[str]] = {}
        for (table_name,) in table_rows:
            info_rows = conn.execute(
                f"PRAGMA table_info('{table_name}')"
            ).fetchall()
            result[table_name] = [row["name"] for row in info_rows]
        return result
    finally:
        conn.close()


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import sys

    # 加载 .env 文件中的环境变量（若存在）
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    def _self_test(db_id: str) -> None:
        """对指定 db_id 执行完整的自测流程。"""
        print(f"=== db_utils 自测: db_id={db_id!r} ===\n")

        # 1. 测试 get_schema
        try:
            schema = get_schema(db_id)
            print(f"--- Schema ({db_id}) ---")
            print(schema)
        except FileNotFoundError as e:
            print(f"[SKIP] get_schema 失败: {e}\n")
            return

        # 2. 测试 get_schema_dict
        schema_dict = get_schema_dict(db_id)
        print(f"\n--- Schema Dict ({db_id}) ---")
        for table_name, columns in schema_dict.items():
            print(f"  {table_name}: {columns}")

        # 3. 测试 execute_sql — 正常查询
        first_table = next(iter(schema_dict), None)
        if first_table:
            print(f"\n--- execute_sql: SELECT * FROM {first_table} LIMIT 3 ---")
            result = execute_sql(db_id, f"SELECT * FROM {first_table} LIMIT 3")
            if result["success"]:
                print(f"  columns: {result['columns']}")
                for row in result["rows"]:
                    print(f"  {row}")
            else:
                print(f"  [FAIL] {result['error']}")

        # 4. 测试 execute_sql — 拒绝写操作
        print(f"\n--- execute_sql: 拒绝 INSERT 语句 ---")
        result = execute_sql(db_id, "INSERT INTO t VALUES (1)")
        print(f"  success={result['success']}, error={result['error']}")

        # 5. 测试 execute_sql — 无效 SQL
        print(f"\n--- execute_sql: 无效 SQL ---")
        result = execute_sql(db_id, "SELECT * FROM nonexistent_table")
        print(f"  success={result['success']}, error={result['error']}")

        # 6. 测试 WITH 子句
        print(f"\n--- execute_sql: WITH 子句 ---")
        if first_table:
            result = execute_sql(
                db_id,
                f"WITH t AS (SELECT * FROM {first_table} LIMIT 1) SELECT * FROM t",
            )
            print(f"  success={result['success']}, columns={result['columns']}")

        print(f"\n=== 自测完成 ===")

    # 尝试 concert_singer，若不存在则回退到已有的 test 库
    test_db_ids = ["concert_singer", "test"]
    tested = False
    for db_id in test_db_ids:
        root = os.environ.get("SPIDER_DB_ROOT", "./spider/database")
        db_path = Path(root) / db_id / f"{db_id}.sqlite"
        if db_path.is_file():
            _self_test(db_id)
            tested = True
            break
        else:
            print(f"[INFO] 数据库 {db_id!r} 在 {db_path.resolve()} 不存在，尝试下一个...\n")

    if not tested:
        print(
            "所有候选数据库均未找到。请确认 SPIDER_DB_ROOT 下存在 "
            f"concert_singer/concert_singer.sqlite 或 test/test.sqlite",
            file=sys.stderr,
        )
        sys.exit(1)
