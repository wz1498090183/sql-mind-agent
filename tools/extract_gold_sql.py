"""
从 Spider train_spider.json 中提取指定 db_id 的 gold_sql，
输出为 eval_set.json 兼容格式，用于 GRPO 对齐训练集的生成。

用法:
    python tools/extract_gold_sql.py --db department_store --limit 10 --output gold_dept_store.json
    python tools/extract_gold_sql.py --db hospital_1 --sample 20
    python tools/extract_gold_sql.py --db store_1 --difficulty hard
"""

import argparse
import json
import random
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 默认 Spider 训练数据路径
_DEFAULT_SPIDER_TRAIN = _project_root / "spider_data" / "train_spider.json"


def load_spider_train(path: str) -> list[dict]:
    """加载 Spider train_spider.json。

    Args:
        path: JSON 文件路径。

    Returns:
        list[dict]: 所有训练条目。
    """
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"Spider 训练数据文件不存在: {filepath.resolve()}")

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("train_spider.json 顶层必须是数组")

    return data


def estimate_difficulty(sql: str) -> str:
    """根据 SQL 复杂度估算难度等级（easy / medium / hard / extra）。

    规则:
        - 多表 JOIN (>= 3 表) 且含 HAVING/嵌套子查询 → extra
        - 含 JOIN 且含 GROUP BY → hard
        - 含 WHERE 子查询或 ORDER BY → medium
        - 其余 → easy

    Args:
        sql: SQL 查询语句。

    Returns:
        str: 难度标签。
    """
    sql_upper = sql.upper()
    has_join = "JOIN" in sql_upper
    has_group_by = "GROUP BY" in sql_upper
    has_having = "HAVING" in sql_upper
    has_subquery = "SELECT" in sql_upper[7:] if len(sql_upper) > 7 else False  # 粗糙检测嵌套
    has_order_by = "ORDER BY" in sql_upper
    has_multiple_joins = sql_upper.count("JOIN") >= 2

    # 统计表引用数量（简单启发式：统计 FROM 和 JOIN 后的标识符）
    table_count = sql_upper.count("JOIN") + 1

    if (table_count >= 3 and has_having) or (has_join and has_subquery and has_group_by):
        return "extra"
    elif has_join and has_group_by:
        return "hard"
    elif has_join or has_subquery or has_order_by:
        return "medium"
    else:
        return "easy"


def extract_gold_sql(
    data: list[dict],
    db_id: str,
    difficulty: str | None = None,
    limit: int | None = None,
    sample: int | None = None,
    random_seed: int = 42,
) -> list[dict[str, str]]:
    """从 Spider 训练数据中提取指定 db_id 的 gold_sql。

    Args:
        data: 所有 Spider 训练条目。
        db_id: 目标数据库标识符。
        difficulty: 按难度过滤（easy/medium/hard/extra），None 表示不过滤。
        limit: 最多返回条数（取前 N 条）。
        sample: 随机采样 N 条（与 limit 互斥，采样优先）。
        random_seed: 随机种子。

    Returns:
        list[dict]: eval_set.json 兼容格式的用例列表。
    """
    # 按 db_id 过滤
    filtered = [item for item in data if item.get("db_id") == db_id]

    if not filtered:
        print(f"[警告] 未找到 db_id={db_id!r} 的 gold_sql。"
              f"Spider 训练数据中共有 {len(data)} 条，"
              f"涉及 {len(set(x.get('db_id', '') for x in data))} 个不同的 db_id。")
        return []

    # 按难度过滤
    if difficulty:
        before = len(filtered)
        filtered = [
            item for item in filtered
            if estimate_difficulty(item.get("query", "")) == difficulty
        ]
        print(f"  按 difficulty={difficulty!r} 过滤: {len(filtered)} 条（原 {before} 条）")

    print(f"  找到 {db_id!r} 共 {len(filtered)} 条 gold_sql")

    # 转换为 eval_set 兼容格式
    result = []
    for item in filtered:
        result.append({
            "question": item.get("question", ""),
            "db_id": item.get("db_id", db_id),
            "gold_sql": item.get("query", ""),
        })

    # 难度分布统计
    difficulty_counts = {}
    for item in filtered:
        d = estimate_difficulty(item.get("query", ""))
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    print(f"  难度分布: {dict(sorted(difficulty_counts.items()))}")

    # 采样
    if sample and sample < len(result):
        rng = random.Random(random_seed)
        result = rng.sample(result, sample)
        print(f"  随机采样: {len(result)} 条（seed={random_seed}）")

    # 截断
    if limit and limit < len(result):
        result = result[:limit]
        print(f"  截断: {len(result)} 条（原 {len(filtered)} 条）")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Spider train_spider.json 提取 gold_sql，生成 eval_set.json 兼容格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python tools/extract_gold_sql.py --db department_store --limit 10\n"
            "  python tools/extract_gold_sql.py --db hospital_1 --sample 20\n"
            "  python tools/extract_gold_sql.py --db department_store --difficulty hard\n"
            "  python tools/extract_gold_sql.py --db department_store --all --output gold_all.json\n"
        ),
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="目标数据库标识符（如 department_store、hospital_1）",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=str(_DEFAULT_SPIDER_TRAIN),
        help=f"Spider train_spider.json 路径（默认: {_DEFAULT_SPIDER_TRAIN}）",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "extra"],
        default=None,
        help="按 SQL 复杂度过滤",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多返回条数（取前 N 条）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="随机采样 N 条",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="导出全部 gold_sql（不限制条数）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认打印到 stdout）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )

    args = parser.parse_args()

    # 加载数据
    print(f"加载 Spider 训练数据: {args.train_file}")
    data = load_spider_train(args.train_file)
    print(f"  共 {len(data)} 条训练条目\n")

    # 提取
    limit = None if args.all else args.limit
    result = extract_gold_sql(
        data=data,
        db_id=args.db,
        difficulty=args.difficulty,
        limit=limit,
        sample=args.sample,
        random_seed=args.seed,
    )

    if not result:
        print("\n未提取到任何 gold_sql 条目。")
        sys.exit(1)

    # 输出
    json_str = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\n已保存 {len(result)} 条到: {output_path.resolve()}")
    else:
        print(f"\n--- 输出 {len(result)} 条 gold_sql ---\n")
        print(json_str)


if __name__ == "__main__":
    main()
