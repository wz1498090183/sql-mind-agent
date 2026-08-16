"""
run_soul_case.py — 同比/环比/多表对比 灵魂 Case 演示脚本。
逐 case 调用 LangGraph 主图节点并抓取中间态，分段打印 ①~⑦，
用于录屏演示和面试展示。

用法:
    python run_soul_case.py              # 运行全部 3 个 case
    python run_soul_case.py --case 1     # 只运行第 1 个 case
    python run_soul_case.py --case 2     # 只运行第 2 个 case
    python run_soul_case.py --case 3     # 只运行第 3 个 case
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.log_utils import logger, new_trace_id
from app.nodes import (
    _topological_levels,
    aggregate_node,
    degrade_node,
    dispatch_node,
    finalize_node,
    plan_node,
    reflect_node,
)
from app.state import SubTask, init_main_state

# ============================================================
# 演示输出美化工具
# ============================================================

# 分隔线宽度
_BAR = "━" * 66

# 颜色 ANSI 码 (Windows 10+ 兼容)
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}

# 尝试启用 Windows 终端 ANSI 支持
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
except Exception:
    pass


def _c(color: str, text: str) -> str:
    """包裹 ANSI 颜色码。"""
    return f"{C.get(color, '')}{text}{C['reset']}"


def _print_header(title: str, emoji: str = "") -> None:
    """打印大标题分隔。"""
    print()
    print(_c("cyan", _BAR))
    print(_c("cyan", f"  {emoji}  {title}"))
    print(_c("cyan", _BAR))
    print()


def _print_section(title: str, emoji: str = "") -> None:
    """打印子标题分隔。"""
    print()
    print(_c("yellow", "━" * 50))
    print(_c("yellow", f"  {emoji} {title}"))
    print(_c("yellow", "━" * 50))


def _print_kv(key: str, value: str, indent: int = 2) -> None:
    """打印键值对。"""
    pad = " " * indent
    print(f"{pad}{_c('bold', key)}: {value}")


def _print_tree(plan: list[SubTask]) -> None:
    """将子任务计划可视化为依赖树。

    使用 Kahn 拓扑分层 + ASCII 树形线渲染依赖关系。

    Args:
        plan: 子任务列表。
    """
    if not plan:
        print("  (无计划)")
        return

    # 拓扑分层
    try:
        levels = _topological_levels(plan)
    except ValueError as e:
        print(f"  {_c('red', f'拓扑排序失败: {e}')}")
        return

    # 构建依赖反向索引：每个子任务被哪些后续子任务依赖
    dependents: dict[str, list[str]] = {}
    for s in plan:
        sid = s["id"]
        if sid not in dependents:
            dependents[sid] = []
        for dep in s.get("depends_on", []):
            if dep not in dependents:
                dependents[dep] = []
            dependents[dep].append(sid)

    # 找出根节点（无依赖的）
    roots = [s for s in plan if not s.get("depends_on", [])]

    # 记录所有已打印节点，避免重复
    printed: set[str] = set()

    def _print_node(sid: str, prefix: str, is_last: bool) -> None:
        """递归打印节点及其依赖子树。"""
        if sid in printed:
            # 交叉引用：用虚线标出
            print(f"{prefix}{'└── ' if is_last else '├── '}{_c('dim', f'[{sid}] (已在上方展示)')}")
            return
        printed.add(sid)

        task = next((s for s in plan if s["id"] == sid), None)
        if not task:
            print(f"{prefix}{'└── ' if is_last else '├── '}[{sid}] (缺失)")
            return

        desc = task.get("description", "-")
        if len(desc) > 48:
            desc = desc[:45] + "..."

        # 状态着色
        status = task.get("status", "pending")
        status_color = {
            "success": "green",
            "failed": "red",
            "pending": "dim",
        }.get(status, "white")
        status_str = _c(status_color, f"[{status}]")

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{_c('bold', f'[{sid}]')} {status_str} {desc}")

        # 打印子节点
        children = dependents.get(sid, [])
        if children:
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(children):
                _print_node(child, child_prefix, i == len(children) - 1)

    # 打印每层的拓扑结构摘要
    print()
    print(f"  {_c('magenta', '拓扑分层')}: ", end="")
    level_summaries = []
    for i, lv in enumerate(levels):
        ids = [s["id"] for s in lv]
        level_summaries.append(f"Layer {i + 1}: {', '.join(ids)}")
    print("  →  ".join(level_summaries))
    print(f"  {_c('dim', f'(共 {len(levels)} 层，层内并行，层间串行)')}")
    print()

    # 打印依赖树
    print(f"  {_c('magenta', '依赖树')}:")
    for i, root in enumerate(roots):
        _print_node(root["id"], "  ", i == len(roots) - 1)
    print()


def _print_subtask_result(task: SubTask) -> None:
    """打印单个子任务的 SQL 和执行结果。

    Args:
        task: 已完成（成功或失败）的子任务。
    """
    sid = task["id"]
    desc = task.get("description", "-")
    status = task.get("status", "?")

    # 状态图标
    icon = {
        "success": _c("green", "✅"),
        "failed": _c("red", "❌"),
    }.get(status, _c("dim", "⏳"))

    print(f"  {icon} {_c('bold', f'[{sid}]')} {desc}")

    if task.get("sql"):
        sql = task["sql"]
        # 美化：对 SQL 中的关键字加粗（简单处理）
        print(f"     {_c('dim', 'SQL:')} {_c('cyan', sql)}")

    if task.get("retry_count", 0) > 0:
        rc = task["retry_count"]
        print(f"     {_c('yellow', '重试次数: ' + str(rc))}")

    if status == "success" and task.get("result"):
        r = task["result"]
        cols = r.get("columns", [])
        rows = r.get("rows", [])
        print(f"     {_c('dim', f'列: {cols}')}")
        print(f"     {_c('dim', f'行数: {len(rows)}')}")
        if rows:
            print(f"     {_c('dim', '数据:')}")
            for row in rows[:8]:  # 最多显示 8 行
                print(f"       {row}")
            if len(rows) > 8:
                print(f"       {_c('dim', f'... 还有 {len(rows) - 8} 行')}")

    if task.get("error"):
        err = task["error"]
        print(f"     {_c('red', '错误: ' + str(err))}")

    print()


def _print_case_banner(case_num: int, case_name: str, difficulty: str) -> None:
    """打印 Case 横幅。"""
    print()
    print(_c("magenta", "╔" + "═" * 64 + "╗"))
    print(_c("magenta", "║")
          + _c("bold", f"  Case {case_num}: {case_name}").ljust(63)
          + _c("magenta", "║"))
    print(_c("magenta", "║")
          + _c("dim", f"  难度: {difficulty}").ljust(63)
          + _c("magenta", "║"))
    print(_c("magenta", "╚" + "═" * 64 + "╝"))
    print()


def _print_reflection_detail(reflection: dict) -> None:
    """打印反思审查详情。

    Args:
        reflection: 反思结果字典 {"passed": bool, "reason": str, "fix_hint": str}。
    """
    passed = reflection.get("passed", False)
    reason = reflection.get("reason", "")
    fix_hint = reflection.get("fix_hint", "")

    verdict = _c("green", "✅ 通过 — 答案可信") if passed else _c("red", "❌ 不通过 — 需要修正")
    print(f"  审查结论: {verdict}")

    if reason:
        reason_color = "green" if reason.upper() == "OK" else "red"
        print(f"  原因: {_c(reason_color, reason)}")

    if fix_hint:
        print(f"  修正建议: {_c('yellow', fix_hint)}")

    print()


# ============================================================
# 单 Case 执行引擎
# ============================================================

def _suppress_loguru_console() -> list:
    """临时关闭 loguru 控制台输出，返回被静音的 handler 列表供恢复。"""
    silenced = []
    for handler in logger._core.handlers.values():
        try:
            # loguru 中 sys.stdout sink 的 _sink 为 <stdout>
            sink_repr = repr(handler._sink)
            if "_io.TextIOWrapper" in sink_repr or "stdout" in sink_repr.lower():
                handler._levelno = 100  # 高于所有日志级别，等于关闭
                silenced.append(handler)
        except Exception:
            pass
    return silenced


def _restore_loguru_console(silenced: list) -> None:
    """恢复 loguru 控制台输出。"""
    for handler in silenced:
        try:
            handler._levelno = 0  # 恢复最低级别
        except Exception:
            pass


def run_single_case(
    case_name: str,
    question: str,
    db_id: str,
    max_iteration: int = 2,
) -> None:
    """执行单个演示 Case 的完整流程。

    流程: ①原始问题 → ②任务规划 → ③子任务执行 → ④聚合计算
           → ⑤反思审查 → ⑥回退对比(若发生) → ⑦最终答案

    Args:
        case_name: Case 名称。
        question: 用户自然语言问题。
        db_id: 数据库标识符。
        max_iteration: 最大反思迭代轮次。
    """
    tid = new_trace_id()

    # 抑制 loguru 控制台输出，保持演示界面整洁
    silenced_handlers = _suppress_loguru_console()

    try:
        # ---- 初始化 ----
        state = init_main_state(
            question=question,
            db_id=db_id,
            trace_id=tid,
            max_iteration=max_iteration,
        )

        # ============================================================
        # ① 原始问题
        # ============================================================
        _print_header(f"原始问题 — {case_name}", "📋")
        print(f"  {_c('bold', '问题')}: {question}")
        print(f"  {_c('bold', '数据库')}: {db_id}")
        print(f"  {_c('bold', '链路 ID')}: {tid}")
        print(f"  {_c('bold', '最大迭代')}: {max_iteration} 轮")

        # ============================================================
        # ② 任务规划
        # ============================================================
        _print_section("任务规划 — LLM 拆解复杂问题为子任务 DAG", "🧩")

        t0 = time.perf_counter()
        plan_result = plan_node(state)
        plan_elapsed = (time.perf_counter() - t0) * 1000
        state.update(plan_result)

        plan: list[SubTask] = state.get("plan", [])
        print(f"  {_c('dim', f'规划耗时: {plan_elapsed:.0f}ms')}")
        print(f"  {_c('dim', f'子任务数: {len(plan)}')}")
        _print_tree(plan)

        # ============================================================
        # ③ 子任务执行
        # ============================================================
        _print_section("子任务执行 — Dispatch 按拓扑层并行调度", "⚙️")

        t0 = time.perf_counter()
        dispatch_result = dispatch_node(state)
        dispatch_elapsed = (time.perf_counter() - t0) * 1000
        state.update(dispatch_result)

        completed: dict[str, SubTask] = state.get("completed", {})
        print(f"  {_c('dim', f'调度总耗时: {dispatch_elapsed:.0f}ms')}")
        print()
        for sid in [s["id"] for s in plan]:
            task = completed.get(sid)
            if task:
                _print_subtask_result(task)
            else:
                print(f"  {_c('red', f'❌ [{sid}] 未找到执行结果')}")

        # ============================================================
        # ④ 聚合计算
        # ============================================================
        _print_section("聚合计算 — LLM 汇总子任务结果 + 预计算数值", "📊")

        t0 = time.perf_counter()
        agg_result = aggregate_node(state)
        agg_elapsed = (time.perf_counter() - t0) * 1000
        state.update(agg_result)

        # 显示预计算摘要
        from app.nodes import _pre_calc_summary
        calc = _pre_calc_summary(completed)
        if calc:
            print(f"  {_c('magenta', '预计算摘要 (自动检测单值数值做环比/差值)')}:")
            for line in calc.split("\n"):
                print(f"    {line}")
            print()

        print(f"  {_c('dim', f'聚合耗时: {agg_elapsed:.0f}ms')}")
        aggregated = state.get("aggregated_answer", "")
        print(f"  {_c('bold', '聚合答案')}:")
        # 自动换行打印
        for line in aggregated.replace("。", "。\n       ").split("\n"):
            print(f"    {line.strip()}")
        print()

        # ============================================================
        # ⑤ 反思审查
        # ============================================================
        _print_section("反思审查 — 三维修语义/逻辑/可执行性校验", "🔍")

        t0 = time.perf_counter()
        reflect_result = reflect_node(state)
        reflect_elapsed = (time.perf_counter() - t0) * 1000
        state.update(reflect_result)

        print(f"  {_c('dim', f'审查耗时: {reflect_elapsed:.0f}ms')}")
        _print_reflection_detail(state.get("reflection", {}))

        # ============================================================
        # ⑥ 回退重试循环 (若发生)
        # ============================================================
        iteration = 0
        retry_history: list[dict] = []  # 记录每轮迭代状态供对比
        max_retries = state.get("max_iteration", 2)

        # 记录首轮状态
        retry_history.append({
            "round": 0,
            "plan": [s.copy() for s in plan],
            "completed": {k: {
                "id": v["id"], "description": v.get("description", ""),
                "status": v.get("status", ""), "sql": v.get("sql", ""),
                "error": v.get("error", ""),
                "result": v.get("result"),
            } for k, v in completed.items()},
            "aggregated_answer": aggregated,
            "reflection": state.get("reflection", {}),
        })

        while (
            not state["reflection"].get("passed", False)
            and iteration < max_retries - 1
        ):
            iteration += 1
            round_label = f"回退重试 — 第 {iteration} 轮迭代对比"
            _print_section(round_label, "🔄")

            # 打印本轮与上轮的对比
            prev = retry_history[-1]
            prev_reflection = prev.get("reflection", {})
            print(f"  {_c('red', '上一轮失败原因')}: {prev_reflection.get('reason', '?')}")
            print(f"  {_c('yellow', 'fix_hint')}: {prev_reflection.get('fix_hint', '?')}")
            print()

            # 更新 iteration 并触发 retry 预处理
            state["iteration"] = state.get("iteration", 0) + 1
            # 清空中间态（模拟 _on_retry）
            state["plan"] = []
            state["completed"] = {}
            state["aggregated_answer"] = None

            # ---- 重新规划 ----
            print(f"  {_c('magenta', '🔄 重新规划 (plan_node 读取 fix_hint)...')}")
            t0 = time.perf_counter()
            plan_result = plan_node(state)
            retry_plan_elapsed = (time.perf_counter() - t0) * 1000
            state.update(plan_result)

            new_plan = state.get("plan", [])
            print(f"  {_c('dim', f'重规划耗时: {retry_plan_elapsed:.0f}ms')}")
            print(f"  {_c('dim', f'新子任务数: {len(new_plan)}')}")

            # 对比新旧计划
            prev_plan = prev.get("plan", [])
            if prev_plan:
                print(f"  {_c('bold', '计划对比')}:")
                prev_ids = {s["id"]: s.get("description", "") for s in prev_plan}
                new_ids = {s["id"]: s.get("description", "") for s in new_plan}
                all_ids = sorted(set(list(prev_ids.keys()) + list(new_ids.keys())))
                for sid in all_ids:
                    prev_desc = prev_ids.get(sid, _c("red", "(已移除)"))
                    new_desc = new_ids.get(sid, _c("red", "(已移除)"))
                    changed = prev_desc != new_desc
                    mark = _c("yellow", " ★ 已修正") if changed else ""
                    if changed:
                        print(f"     [{sid}]: {_c('dim', prev_desc)} → {_c('green', new_desc)}{mark}")
                    else:
                        print(f"     [{sid}]: {new_desc}")

            _print_tree(new_plan)

            # ---- 重新调度 ----
            print(f"  {_c('magenta', '🔄 重新调度执行...')}")
            t0 = time.perf_counter()
            dispatch_result = dispatch_node(state)
            retry_dispatch_elapsed = (time.perf_counter() - t0) * 1000
            state.update(dispatch_result)

            new_completed = state.get("completed", {})
            print(f"  {_c('dim', f'重调度耗时: {retry_dispatch_elapsed:.0f}ms')}")
            print()

            # 对比结果变化
            prev_completed = prev.get("completed", {})
            for sid in [s["id"] for s in new_plan]:
                task = new_completed.get(sid)
                if not task:
                    continue
                prev_task = prev_completed.get(sid, {})
                prev_status = prev_task.get("status", "?") if prev_task else "?"
                cur_status = task.get("status", "?")
                status_changed = prev_status != cur_status
                change_mark = (
                    f"  {_c('red', '← 上一轮')}: {prev_status}"
                    if status_changed else ""
                )
                if status_changed:
                    print(f"  [{sid}] {_c('green', '状态改善')}: {prev_status} → {cur_status}{change_mark}")
                _print_subtask_result(task)

            # ---- 重新聚合 ----
            print(f"  {_c('magenta', '🔄 重新聚合...')}")
            t0 = time.perf_counter()
            agg_result = aggregate_node(state)
            state.update(agg_result)

            retry_aggregated = state.get("aggregated_answer", "")
            print(f"  {_c('bold', '新聚合答案')}:")
            for line in retry_aggregated.replace("。", "。\n       ").split("\n"):
                print(f"    {line.strip()}")
            print()

            # ---- 重新反思 ----
            print(f"  {_c('magenta', '🔄 重新反思审查...')}")
            t0 = time.perf_counter()
            reflect_result = reflect_node(state)
            state.update(reflect_result)

            _print_reflection_detail(state.get("reflection", {}))

            # 记录本轮
            retry_history.append({
                "round": iteration,
                "plan": [s.copy() for s in new_plan],
                "completed": {k: {
                    "id": v["id"], "description": v.get("description", ""),
                    "status": v.get("status", ""), "sql": v.get("sql", ""),
                    "error": v.get("error", ""),
                    "result": v.get("result"),
                } for k, v in new_completed.items()},
                "aggregated_answer": retry_aggregated,
                "reflection": state.get("reflection", {}),
            })

        # ============================================================
        # ⑦ 最终答案
        # ============================================================
        passed = state["reflection"].get("passed", False)

        if passed:
            result = finalize_node(state)
            state.update(result)
            verdict = _c("green", "✅ 审查通过 — 答案可信")
        else:
            result = degrade_node(state)
            state.update(result)
            verdict = _c("yellow", "⚠️ 审查未通过 — 降级回答")

        _print_header("最终答案", "✅")

        print(f"  状态: {verdict}")
        print(f"  总迭代轮次: {iteration + 1}")
        print()

        final_answer = state.get("final_answer", state.get("aggregated_answer", "(无答案)"))
        print(f"  {_c('bold', '答案内容')}:")
        print(f"  {'─' * 60}")
        for line in final_answer.replace("。", "。\n       ").split("\n"):
            print(f"  {line.strip()}")
        print(f"  {'─' * 60}")
        print()

        # 总结统计
        _print_section("执行统计", "📈")
        total_tasks = len(plan)
        success_tasks = len([
            t for t in completed.values() if t.get("status") == "success"
        ])
        failed_tasks = total_tasks - success_tasks
        print(f"  子任务总数: {total_tasks}")
        print(f"  {_c('green', f'成功: {success_tasks}')}")
        if failed_tasks > 0:
            print(f"  {_c('red', f'失败: {failed_tasks}')}")
        print(f"  反思迭代: {iteration + 1} 轮")
        print(f"  最终状态: {verdict}")
        print()

    finally:
        # 恢复 loguru 控制台输出
        _restore_loguru_console(silenced_handlers)


# ============================================================
# 内置 3 个灵魂 Case
# ============================================================

SOUL_CASES: list[dict[str, Any]] = [
    # -------------------------------------------------------
    # Case 1: 多表对比 — 客户 + 订单 + 产品 复合分析
    # -------------------------------------------------------
    {
        "name": "多表对比 — 客户/订单/产品 三维分析",
        "question": (
            "department_store 数据库有哪些客户？总共有多少客户？"
            "有多少个订单？平均每个客户下多少单？"
            "列出产品类型（product_type_code）及其产品数量。"
        ),
        "db_id": "department_store",
        "difficulty": "⭐⭐（中等）",
    },

    # -------------------------------------------------------
    # Case 2: 环比分析 — 价格对比、类型占比
    # -------------------------------------------------------
    {
        "name": "同比环比 — 产品类型价格对比与占比分析",
        "question": (
            "department_store 数据库的 Products 表中，"
            "Hardware 类型和 Clothes 类型的产品各有多少？"
            "哪种类型产品数量更多？"
            "Hardware 产品的平均价格是多少？"
            "Clothes 产品的平均价格是多少？"
            "两种类型的平均价格相差多少？相差百分比是多少？"
        ),
        "db_id": "department_store",
        "difficulty": "⭐⭐⭐（困难）",
    },

    # -------------------------------------------------------
    # Case 3: 多表关联 + 依赖链 — 客户订单统计 + 深度钻取
    # -------------------------------------------------------
    {
        "name": "多表关联+依赖 — 客户订单统计 + 深度钻取地址",
        "question": (
            "department_store 数据库中，哪些客户下过订单？"
            "按客户统计订单数量，列出订单最多的前 3 名客户及其订单数。"
            "另外找出订单最多的那个客户，"
            "查询其在 Customer_Addresses 表中的详细地址信息（address_details, city）。"
        ),
        "db_id": "department_store",
        "difficulty": "⭐⭐⭐（困难）",
    },
]


# ============================================================
# 主入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL Mind Agent — 灵魂 Case 演示脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run_soul_case.py               # 运行全部 3 个 case\n"
            "  python run_soul_case.py --case 1      # 只运行第 1 个 case\n"
            "  python run_soul_case.py --case 2      # 只运行第 2 个 case\n"
            "  python run_soul_case.py --case 1 2    # 运行第 1、2 个 case\n"
        ),
    )
    parser.add_argument(
        "--case", "-c",
        type=int,
        nargs="+",
        default=None,
        help="指定要运行的 Case 编号（1~3），不指定则运行全部",
    )
    parser.add_argument(
        "--max-iteration", "-m",
        type=int,
        default=2,
        help="最大反思迭代轮次（默认 2）",
    )

    args = parser.parse_args()

    # 筛选要运行的 Case
    if args.case:
        selected = []
        for c in args.case:
            if 1 <= c <= len(SOUL_CASES):
                selected.append(SOUL_CASES[c - 1])
            else:
                print(
                    f"警告: Case {c} 不存在（有效范围 1~{len(SOUL_CASES)}），已跳过",
                    file=sys.stderr,
                )
        if not selected:
            print("错误: 没有有效的 Case 可运行", file=sys.stderr)
            sys.exit(1)
    else:
        selected = SOUL_CASES

    # ============================================================
    # 开场 Banner
    # ============================================================
    print()
    print(_c("magenta", "╔" + "═" * 64 + "╗"))
    print(_c("magenta", "║")
          + _c("bold", "  🚀 SQL Mind Agent — 灵魂 Case 演示").ljust(63)
          + _c("magenta", "║"))
    print(_c("magenta", "║")
          + _c("dim", "  Text2SQL 多步骤智能问答 Agent").ljust(63)
          + _c("magenta", "║"))
    print(_c("magenta", "║")
          + _c("dim", f"  共 {len(selected)} 个 Case，最大迭代 {args.max_iteration} 轮").ljust(63)
          + _c("magenta", "║"))
    print(_c("magenta", "╚" + "═" * 64 + "╝"))
    print()

    # 逐个执行
    total_start = time.perf_counter()
    for case in selected:
        case_num = SOUL_CASES.index(case) + 1
        _print_case_banner(case_num, case["name"], case["difficulty"])
        run_single_case(
            case_name=case["name"],
            question=case["question"],
            db_id=case["db_id"],
            max_iteration=args.max_iteration,
        )

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # ============================================================
    # 收尾 Banner
    # ============================================================
    print()
    print(_c("green", _BAR))
    print(_c("green", f"  🎉 全部 {len(selected)} 个 Case 执行完毕！"))
    print(_c("green", f"  总耗时: {total_elapsed:.0f}ms ({total_elapsed/1000:.1f}s)"))
    print(_c("green", _BAR))
    print()


if __name__ == "__main__":
    main()
