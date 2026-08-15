"""
prepare_data.py — 从 Spider 提取复杂 SQL 子集，转成 LLaMA-Factory Alpaca 格式。

流程:
  1. 读 train_spider.json，筛选复杂 SQL（JOIN/子查询/GROUP BY+HAVING/时间日期）
  2. 从 tables.json 重建每个 db 的 CREATE TABLE DDL 作为 Schema 上下文
  3. 每条转成:
       {
         "instruction": "根据数据库Schema回答问题，只输出SQL。\nSchema:\n{DDL}\n问题:{question}",
         "input": "",
         "output": "{gold_sql}"
       }
  4. 输出 train/spider_complex.json
  5. 打印 LLaMA-Factory dataset_info.json 注册片段

用法:
    python train/prepare_data.py
    python train/prepare_data.py --max-samples 1000
    python train/prepare_data.py --db-filter department_store hospital_1
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# 项目根目录
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 默认路径（spider_data 在仓库根，train 输出在 grpo-train/train 下）
_DEFAULT_TRAIN_SPIDER = _project_root.parent / "spider_data" / "train_spider.json"
_DEFAULT_TABLES_JSON = _project_root.parent / "spider_data" / "tables.json"
_DEFAULT_OUTPUT = _project_root / "train" / "spider_complex.json"

# Spider type → SQLite type 映射
_TYPE_MAP = {
    "text": "TEXT",
    "number": "INTEGER",
    "time": "TEXT",
    "boolean": "INTEGER",
}


def _map_type(spider_type: str) -> str:
    """将 Spider 类型映射为 SQLite 类型。"""
    return _TYPE_MAP.get(spider_type.lower(), "TEXT")


def _sanitize_name(name: str) -> str:
    """清理列名/表名，去除空格和特殊字符。"""
    return name.strip().replace(" ", "_").replace("-", "_")


def build_ddl(db_schema: dict) -> str:
    """从 tables.json 条目重建 CREATE TABLE DDL。

    Spider tables.json 结构:
      - table_names_original: 原始表名列表
      - column_names_original: [[table_idx, col_name], ...]，table_idx=-1 表示 *
      - column_types: 对应的 Spider 类型列表
      - primary_keys: 主键列索引列表
      - foreign_keys: [[col_idx, ref_col_idx], ...]

    Args:
        db_schema: tables.json 中单条数据库的 schema 定义。

    Returns:
        str: 完整的 CREATE TABLE DDL，表间用空行分隔。
    """
    tables = db_schema["table_names_original"]
    columns = db_schema["column_names_original"]
    types = db_schema["column_types"]
    pks = set(db_schema.get("primary_keys", []))
    fks = db_schema.get("foreign_keys", [])

    # 按表索引分组列
    table_columns: dict[int, list[tuple[int, str, str]]] = {}
    # col_idx → (col_name, col_type, global_col_idx)
    for col_idx, (table_idx, col_name) in enumerate(columns):
        if table_idx == -1:
            continue  # 跳过 * 行
        if table_idx not in table_columns:
            table_columns[table_idx] = []
        table_columns[table_idx].append(
            (col_idx, _sanitize_name(col_name), _map_type(types[col_idx]))
        )

    # 构建 FK 反向索引: ref_col_idx → (col_idx, ref_col_idx)
    fk_ref_map: dict[int, list[tuple[int, int]]] = {}
    for col_idx, ref_col_idx in fks:
        if ref_col_idx not in fk_ref_map:
            fk_ref_map[ref_col_idx] = []
        fk_ref_map[ref_col_idx].append((col_idx, ref_col_idx))

    # 找出每个 FK 引用列属于哪个表/哪个列
    def _resolve_fk_ref(ref_col_idx: int) -> tuple[int, str] | None:
        """根据全局列索引解析引用的表索引和列名。"""
        for ti, cols in table_columns.items():
            for ci, name, ctype in cols:
                if ci == ref_col_idx:
                    return ti, name
        return None

    ddl_parts: list[str] = []
    for table_idx in sorted(table_columns.keys()):
        table_name = _sanitize_name(tables[table_idx])
        cols = table_columns[table_idx]

        lines = [f"CREATE TABLE `{table_name}` ("]
        col_defs: list[str] = []
        pk_cols: list[str] = []
        fk_defs: list[str] = []

        for col_idx, col_name, col_type in cols:
            parts = [f"`{col_name}`", col_type]
            if col_idx in pks:
                pk_cols.append(f"`{col_name}`")
            col_defs.append("  " + " ".join(parts))

        # 主键
        if pk_cols:
            col_defs.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

        # 外键
        for col_idx, ref_col_idx in fks:
            # 找 col_idx 属于哪个表
            for ti, ci_name, _ in table_columns.get(table_idx, []):
                if ti == table_idx and col_idx == next(
                    (_ci for _ci, _cn, _ct in table_columns[table_idx] if _ci == col_idx),
                    None,
                ):
                    break
            ref_info = _resolve_fk_ref(ref_col_idx)
            if ref_info:
                ref_ti, ref_col = ref_info
                ref_table_name = _sanitize_name(tables[ref_ti])
                # 找 col_idx 对应的列名
                col_name = next(
                    (_cn for _ci, _cn, _ in table_columns[table_idx] if _ci == col_idx),
                    None,
                )
                if col_name:
                    fk_defs.append(
                        f"  FOREIGN KEY (`{col_name}`) "
                        f"REFERENCES `{ref_table_name}`(`{ref_col}`)"
                    )

        # 合并 FK 定义
        col_defs.extend(fk_defs)

        # 去除最后多余的逗号
        lines.append(",\n".join(col_defs))
        lines.append(");")
        ddl_parts.append("\n".join(lines))

    return "\n\n".join(ddl_parts)


def is_complex(item: dict) -> bool:
    """判定 SQL 是否属于复杂类别。

    命中以下任一规则即算复杂:
      1. SQL 含 JOIN（多表连接）
      2. 含子查询 / 嵌套 SELECT
      3. 含 GROUP BY + HAVING
      4. 含时间/日期对比相关关键词

    Args:
        item: train_spider.json 中的单条训练条目。

    Returns:
        bool: 是否为复杂 SQL。
    """
    sql = item.get("query", "")
    sql_upper = sql.upper()

    # 规则 1: JOIN
    if "JOIN" in sql_upper:
        return True

    # 规则 2: 子查询/嵌套 SELECT（至少两个 SELECT）
    if sql_upper.count("SELECT") >= 2:
        return True

    # 规则 3: GROUP BY + HAVING
    if "GROUP BY" in sql_upper and "HAVING" in sql_upper:
        return True

    # 规则 4: 时间/日期对比关键词
    time_keywords = [
        "BETWEEN", "DATE", "YEAR", "MONTH", "STRFTIME", "JULIANDAY",
        "DATETIME", "INTERVAL", "AGE",
    ]
    if any(kw in sql_upper for kw in time_keywords):
        return True

    return False


def load_spider_train(path: str) -> list[dict]:
    """加载 Spider train_spider.json。

    Args:
        path: JSON 文件路径。

    Returns:
        list[dict]: 所有训练条目。
    """
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"训练数据文件不存在: {filepath.resolve()}")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("train_spider.json 顶层必须是数组")
    return data


def load_tables_json(path: str) -> dict[str, dict]:
    """加载 Spider tables.json，返回以 db_id 为键的 schema 字典。

    Args:
        path: tables.json 文件路径。

    Returns:
        dict[str, dict]: {db_id: schema_dict}。
    """
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"tables.json 不存在: {filepath.resolve()}")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("tables.json 顶层必须是数组")
    return {item["db_id"]: item for item in data}


def prepare_data(
    train_data: list[dict],
    tables_map: dict[str, dict],
    max_samples: int | None = None,
    db_filter: list[str] | None = None,
) -> list[dict]:
    """从 Spider 训练数据中提取复杂 SQL 并转为 LLaMA-Factory Alpaca 格式。

    Args:
        train_data: train_spider.json 的全部条目。
        tables_map: {db_id: schema_dict} 映射。
        max_samples: 最大输出条数，None 表示不限制。
        db_filter: 仅包含指定 db_id 的样本，None 表示全部。

    Returns:
        list[dict]: Alpaca 格式的训练数据。
    """
    # 筛选复杂 SQL
    complex_samples = [item for item in train_data if is_complex(item)]
    print(f"复杂 SQL 总数: {len(complex_samples)}（原始 {len(train_data)} 条）")

    # 按 db_id 过滤
    if db_filter:
        filter_set = set(db_filter)
        complex_samples = [
            item for item in complex_samples if item["db_id"] in filter_set
        ]
        print(f"  按 db_filter 过滤后: {len(complex_samples)} 条")

    # 截断
    if max_samples and len(complex_samples) > max_samples:
        complex_samples = complex_samples[:max_samples]
        print(f"  截断至: {len(complex_samples)} 条")

    # 转换格式
    result: list[dict] = []
    skipped_no_schema: int = 0
    schema_cache: dict[str, str] = {}  # db_id → DDL 缓存

    for item in complex_samples:
        db_id = item["db_id"]

        # 获取/缓存该 db 的 DDL
        if db_id not in schema_cache:
            schema = tables_map.get(db_id)
            if schema is None:
                skipped_no_schema += 1
                continue
            schema_cache[db_id] = build_ddl(schema)

        ddl = schema_cache[db_id]
        question = item["question"]
        gold_sql = item["query"]

        instruction = (
            f"根据数据库Schema回答问题，只输出SQL。\n"
            f"Schema:\n{ddl}\n"
            f"问题:{question}"
        )

        result.append({
            "instruction": instruction,
            "input": "",
            "output": gold_sql,
            "db_id": db_id,
        })

    if skipped_no_schema:
        print(f"  跳过 {skipped_no_schema} 条（tables.json 中无 schema 定义）")

    print(f"最终输出: {len(result)} 条")
    return result


def print_dataset_info_json(
    output_path: str,
    dataset_name: str = "spider_complex",
) -> None:
    """打印需追加到 LLaMA-Factory data/dataset_info.json 的 JSON 片段。

    Args:
        output_path: 训练数据文件的绝对路径。
        dataset_name: 数据集注册名称。
    """
    fragment = {
        dataset_name: {
            "file_name": output_path,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "output": "output",
            },
        },
    }
    print()
    print("=" * 60)
    print("  请将以下片段追加到 LLaMA-Factory 的 data/dataset_info.json 中:")
    print("=" * 60)
    print(json.dumps(fragment, ensure_ascii=False, indent=2))
    print("=" * 60)


def print_preview(result: list[dict], n: int = 3) -> None:
    """打印前 N 条转换结果预览。

    Args:
        result: 转换后的 Alpaca 格式数据。
        n: 预览条数。
    """
    print()
    print("=" * 60)
    print(f"  前 {min(n, len(result))} 条转换结果预览:")
    print("=" * 60)
    for i, item in enumerate(result[:n]):
        instruction = item["instruction"]
        output = item["output"]
        # 截断过长的 Schema 用于预览
        if len(instruction) > 600:
            instruction = instruction[:600] + "\n  ...(Schema截断)"
        print(f"\n--- 第 {i + 1} 条 ---")
        print(f"instruction:\n{instruction}")
        print(f"output: {output}")


def print_statistics(result: list[dict], complex_samples: list[dict]) -> None:
    """打印筛选统计信息。

    Args:
        result: 最终输出数据。
        complex_samples: 筛选出的复杂样本（截断前）。
    """
    # 复杂度规则统计
    stats = {
        "含 JOIN": 0,
        "含子查询(嵌套SELECT)": 0,
        "含 GROUP BY + HAVING": 0,
        "含时间/日期关键词": 0,
    }
    for item in complex_samples[: len(result)]:
        sql = item.get("query", "")
        sql_upper = sql.upper()
        if "JOIN" in sql_upper:
            stats["含 JOIN"] += 1
        if sql_upper.count("SELECT") >= 2:
            stats["含子查询(嵌套SELECT)"] += 1
        if "GROUP BY" in sql_upper and "HAVING" in sql_upper:
            stats["含 GROUP BY + HAVING"] += 1
        time_keywords = [
            "BETWEEN", "DATE", "YEAR", "MONTH", "STRFTIME", "JULIANDAY",
            "DATETIME", "INTERVAL", "AGE",
        ]
        if any(kw in sql_upper for kw in time_keywords):
            stats["含时间/日期关键词"] += 1

    # 按 db_id 分布（Top 10）
    db_counter = Counter(item["db_id"] for item in complex_samples[: len(result)])

    print()
    print("=" * 60)
    print("  筛选统计")
    print("=" * 60)
    print(f"  原始样本总数: 7000")
    print(f"  复杂样本总数: {len(complex_samples)}")
    print(f"  最终输出条数: {len(result)}")
    print()
    print("  复杂度规则命中分布（一条可命中多条规则）:")
    for rule, count in stats.items():
        print(f"    {rule}: {count}")
    print()
    print("  按数据库分布（Top 15）:")
    for db_id, count in db_counter.most_common(15):
        print(f"    {db_id}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Spider 提取复杂 SQL 子集，转为 LLaMA-Factory Alpaca 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python train/prepare_data.py\n"
            "  python train/prepare_data.py --max-samples 1000\n"
            "  python train/prepare_data.py --db-filter department_store hospital_1\n"
            "  python train/prepare_data.py --dataset-name spider_complex_v2"
        ),
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="最大输出条数（默认不限制，约 4000+ 条）",
    )
    parser.add_argument(
        "--db-filter",
        type=str,
        nargs="+",
        default=None,
        help="仅包含指定 db_id 的样本（空格分隔，如 department_store hospital_1）",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=str(_DEFAULT_TRAIN_SPIDER),
        help=f"train_spider.json 路径（默认: {_DEFAULT_TRAIN_SPIDER}）",
    )
    parser.add_argument(
        "--tables-file",
        type=str,
        default=str(_DEFAULT_TABLES_JSON),
        help=f"tables.json 路径（默认: {_DEFAULT_TABLES_JSON}）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=f"输出 JSON 文件路径（默认: {_DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="spider_complex",
        help="LLaMA-Factory 数据集注册名称（默认: spider_complex）",
    )

    args = parser.parse_args()

    # 加载数据
    print(f"加载训练数据: {args.train_file}")
    train_data = load_spider_train(args.train_file)
    print(f"  共 {len(train_data)} 条\n")

    print(f"加载 tables.json: {args.tables_file}")
    tables_map = load_tables_json(args.tables_file)
    print(f"  共 {len(tables_map)} 个数据库\n")

    # 筛选 & 转换
    complex_samples = [item for item in train_data if is_complex(item)]
    result = prepare_data(
        train_data=train_data,
        tables_map=tables_map,
        max_samples=args.max_samples,
        db_filter=args.db_filter,
    )

    if not result:
        print("\n[错误] 未生成任何训练数据，请检查筛选条件。")
        sys.exit(1)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(result)} 条到: {output_path.resolve()}")

    # 打印预览
    print_preview(result, n=3)

    # 打印统计
    print_statistics(result, complex_samples)

    # 打印 dataset_info.json 注册片段
    print_dataset_info_json(
        output_path=str(output_path.resolve()),
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
