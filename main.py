"""
命令行入口。
用法:
    python main.py --question "你的问题"
    python main.py --demo
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.graph import build_main_graph
from app.log_utils import get_logger, new_trace_id
from app.state import init_main_state

# ============================================================
# 内置演示问题
# ============================================================
_DEMO_QUESTION = "department_store 数据库的 Customers 表有多少条记录？Products 表有哪些产品名称？"


def _print_section(title: str) -> None:
    """打印分隔标题。"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _run(question: str, db_id: str) -> None:
    """执行完整的 Text2SQL 流程并打印结果。

    Args:
        question: 用户自然语言问题。
        db_id: 目标数据库标识符。
    """
    tid = new_trace_id()
    log = get_logger(tid)

    # 初始化状态
    state = init_main_state(
        question=question,
        db_id=db_id,
        trace_id=tid,
        max_iteration=2,  # 最多反思重试 2 轮
    )

    _print_section("原始问题")
    print(f"  {question}")
    print(f"  数据库: {db_id}")
    print(f"  trace_id: {tid}")

    # 构建图
    graph = build_main_graph()

    # 执行
    log.info("main: 开始 invoke 主图")
    final_state = graph.invoke(state)
    log.info(f"main: invoke 完成  status={final_state.get('status', '?')}")

    # 展示执行计划
    plan = final_state.get("plan", [])
    _print_section("执行计划")
    if plan:
        for s in plan:
            deps = s.get("depends_on", [])
            dep_str = f" (依赖: {', '.join(deps)})" if deps else ""
            print(f"  [{s['id']}] {s['description']}{dep_str}")
    else:
        print("  (无计划)")

    # 展示各子任务 SQL 与结果
    completed = final_state.get("completed", {})
    _print_section("子任务执行详情")
    if completed:
        for sid in plan or completed:
            task = completed.get(sid["id"] if isinstance(sid, dict) else sid)
            if not task:
                print(f"  [{sid if isinstance(sid, str) else sid.get('id', '?')}] 未找到结果")
                continue
            print(f"  [{task['id']}] {task.get('description', '-')}")
            print(f"       状态: {task.get('status', '-')}")
            if task.get("sql"):
                print(f"       SQL:  {task['sql']}")
            if task.get("status") == "success" and task.get("result"):
                r = task["result"]
                print(f"       列:   {r.get('columns', [])}")
                row_count = len(r.get("rows", []))
                if row_count <= 10:
                    print(f"       行数: {row_count}")
                    for row in r.get("rows", []):
                        print(f"             {row}")
                else:
                    print(f"       行数: {row_count}（仅显示前5行）")
                    for row in r.get("rows", [])[:5]:
                        print(f"             {row}")
            if task.get("error"):
                print(f"       错误: {task['error']}")
            print()
    else:
        print("  (无执行结果)")

    # 展示最终答案
    answer = final_state.get("aggregated_answer", "")
    _print_section("最终答案")
    print(f"  {answer}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL Mind Agent — Text2SQL 多步骤智能问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               '  python main.py --question "Customers 表有多少条记录？"\n'
               "  python main.py --demo",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="用户自然语言问题",
    )
    parser.add_argument(
        "--db_id", "-d",
        type=str,
        default="department_store",
        help='目标数据库标识符（默认: department_store）',
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用内置示例问题直接运行",
    )

    args = parser.parse_args()

    if args.demo:
        print(f"[DEMO] 使用内置示例问题: {_DEMO_QUESTION}")
        _run(question=_DEMO_QUESTION, db_id=args.db_id)
    elif args.question:
        _run(question=args.question, db_id=args.db_id)
    else:
        parser.print_help()
        print()
        print("提示: 使用 --demo 运行示例，或 --question 指定你的问题。")


if __name__ == "__main__":
    main()
