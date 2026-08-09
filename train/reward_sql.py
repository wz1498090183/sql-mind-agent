"""
GRPO 强化学习奖励函数模块。
实现三维加权奖励函数，供 LLaMA-Factory GRPO 训练调用。

三维权重（总纲6.4）：
    1. 执行正确性（0.6）— SQL 执行结果是否与 gold_sql 一致
    2. 结构合理性（0.2）— SQL 结构关键词命中比例
    3. 规范合规性（0.2）— 无 SELECT * / 无高危写操作

用法:
    from train.reward_sql import compute_reward, batch_reward

    # 单条评估
    score = compute_reward(generated_sql, gold_sql, db_id)

    # 批量评估（GRPO samples 格式）
    scores = batch_reward(samples)

参考:
    - LLaMA-Factory GRPO 自定义 reward 约定
    - evaluate.py 的结果集合比较逻辑（本模块内联了一份以保持独立）
"""

import re
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import sqlparse
from sqlparse.sql import Statement

from app.db_utils import execute_sql

# ============================================================
# 常量定义
# ============================================================

# 三维权重（总和为 1.0）
_WEIGHT_EXECUTION = 0.6     # 执行正确性权重
_WEIGHT_STRUCTURE = 0.2     # 结构合理性权重
_WEIGHT_COMPLIANCE = 0.2    # 规范合规性权重

# 执行正确性分档
_EXEC_FULL = 1.0            # 结果完全一致
_EXEC_PARTIAL = 0.2         # 可执行但结果不一致（部分给分，鼓励可执行）
_EXEC_FAIL = 0.0            # 语法错 / 不可执行

# 高危写操作关键字集合（命中直接总分归零）
_FORBIDDEN_KEYWORDS: set[str] = {"DROP", "DELETE", "UPDATE", "INSERT"}

# 结构特征关键词及其正则模式（大小写不敏感匹配）
_STRUCTURE_PATTERNS: dict[str, str] = {
    "JOIN": r"\b(JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|OUTER\s+JOIN|CROSS\s+JOIN|FULL\s+JOIN)\b",
    "GROUP_BY": r"\bGROUP\s+BY\b",
    "ORDER_BY": r"\bORDER\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "UNION": r"\bUNION\b",
    "WHERE": r"\bWHERE\b",
    "DISTINCT": r"\bDISTINCT\b",
    "LIMIT": r"\bLIMIT\b",
    "SUBQUERY": r"\(\s*SELECT\b",  # 嵌套子查询
}


# ============================================================
# SQL 结果集合比较 — 内联自 evaluate.py，避免引入 LangGraph 依赖链
# ============================================================
def _normalize_sql_result(result: dict) -> tuple[frozenset[tuple], list[str]] | tuple[None, None]:
    """将 execute_sql 返回的 dict 规范化为可比较的 frozenset。

    对 columns 排序，对 rows 按列索引重排后转 tuple，
    最终返回所有行的 frozenset + 排序后的列名列表，彻底消除列序与行序的影响。

    Args:
        result: execute_sql 返回的结构化 dict。

    Returns:
        tuple[frozenset, list] | (None, None): (规范化行集, 排序后列名)，失败时返回 (None, None)。
    """
    if not result or not result.get("success"):
        return None, None

    columns: list[str] = result.get("columns", [])
    rows: list[tuple] = result.get("rows", [])

    if not columns:
        return frozenset(), []

    # 按列名排序，建立 原始索引 → 排序后索引 的映射
    sorted_cols = sorted(enumerate(columns), key=lambda x: x[1])
    # index_map: 原始列索引 → 排序后列索引
    index_map = [0] * len(columns)
    for new_idx, (orig_idx, _col_name) in enumerate(sorted_cols):
        index_map[orig_idx] = new_idx

    # 对每行按排序后的列顺序重排
    normalized_rows: list[tuple] = []
    for row in rows:
        reordered = [None] * len(columns)
        for orig_idx, value in enumerate(row):
            reordered[index_map[orig_idx]] = value
        normalized_rows.append(tuple(reordered))

    # 返回行集的 frozenset + 排序后的列名
    return frozenset(normalized_rows), [col for _, col in sorted_cols]


def _compare_sql_results(
    gen_result: dict | None,
    gold_result: dict | None,
) -> tuple[bool, str]:
    """比较生成 SQL 与 gold_sql 的执行结果是否一致。

    比较规则（与 evaluate.py 保持一致）:
        1. 两边都失败 → 一致（不是 SQL 问题），返回 True
        2. 一边失败一边成功 → 不一致
        3. 都成功 → 列集相等 + 行集（frozenset）相等

    Args:
        gen_result: 生成 SQL 的执行结果（db_utils.execute_sql 返回值）。
        gold_result: gold_sql 的执行结果。

    Returns:
        tuple[bool, str]: (是否一致, 差异描述)。
    """
    gen_ok = gen_result and gen_result.get("success")
    gold_ok = gold_result and gold_result.get("success")

    # 两边都失败 → 视为一致（数据结构问题，非 SQL 问题）
    if not gen_ok and not gold_ok:
        return True, "两边均执行失败（可能是数据库结构问题）"

    if not gen_ok:
        return False, f"生成 SQL 执行失败: {gen_result.get('error', '未知') if gen_result else '结果为空'}"

    if not gold_ok:
        return False, f"gold_sql 执行失败（评估数据可能有问题）: {gold_result.get('error', '未知') if gold_result else '结果为空'}"

    # 比较列集
    gen_cols = set(gen_result.get("columns", []))
    gold_cols = set(gold_result.get("columns", []))
    if gen_cols != gold_cols:
        return False, (
            f"列集不一致: 生成={sorted(gen_cols)}, Gold={sorted(gold_cols)}"
        )

    # 规范化后比较行集
    gen_rows, _ = _normalize_sql_result(gen_result)
    gold_rows, _gold_cols = _normalize_sql_result(gold_result)

    if gen_rows is None or gold_rows is None:
        return False, "结果规范化失败"

    if gen_rows == gold_rows:
        return True, "结果完全一致"
    else:
        only_gen = gen_rows - gold_rows
        only_gold = gold_rows - gen_rows
        parts: list[str] = []
        if only_gen:
            parts.append(f"仅生成 SQL 有 {len(only_gen)} 行")
        if only_gold:
            parts.append(f"仅 Gold 有 {len(only_gold)} 行")
        return False, " | ".join(parts)


# ============================================================
# 辅助函数
# ============================================================
def _extract_structure_features(sql: str) -> set[str]:
    """从 SQL 文本中提取结构特征关键词集合。

    使用正则匹配识别 JOIN / GROUP BY / HAVING / 子查询等结构特征，
    用于与 gold_sql 的结构做对比打分。

    Args:
        sql: SQL 语句文本。

    Returns:
        set[str]: 命中的结构特征名称集合。
    """
    features: set[str] = set()
    if not sql:
        return features
    upper_sql = sql.upper()
    for name, pattern in _STRUCTURE_PATTERNS.items():
        if re.search(pattern, upper_sql, re.IGNORECASE):
            features.add(name)
    return features


def _check_compliance(sql: str) -> tuple[bool, str]:
    """检查 SQL 的规范合规性。

    检测项：
        1. 是否包含 SELECT *（全表扫描，不鼓励使用）
        2. 是否包含高危写操作关键字（DROP/DELETE/UPDATE/INSERT）

    Args:
        sql: SQL 语句文本。

    Returns:
        tuple[bool, str]: (是否通过合规检查, 违规原因描述)。
            通过返回 (True, "")，不通过返回 (False, 原因)。
    """
    if not sql:
        return False, "SQL 为空"

    upper_sql = sql.upper()

    # 检测 SELECT *（全表扫描，不鼓励使用）
    if re.search(r"\bSELECT\s+\*", upper_sql):
        return False, "包含 SELECT *（全表扫描），不鼓励使用"

    # 检测高危写操作关键字
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            return False, f"包含高危写操作关键字: {keyword}"

    return True, ""


def _print_reward_detail(
    generated_sql: str,
    gold_sql: str,
    db_id: str,
    exec_score: float,
    exec_detail: str,
    struct_score: float,
    struct_detail: str,
    compliance_score: float,
    compliance_detail: str,
    total: float,
) -> None:
    """打印单条奖励明细日志，便于调试奖励是否合理。

    Args:
        generated_sql: 模型生成的 SQL。
        gold_sql: 参考答案 SQL。
        db_id: 数据库标识符。
        exec_score: 执行正确性维度得分（未加权原始分）。
        exec_detail: 执行正确性详细说明。
        struct_score: 结构合理性维度得分（未加权原始分）。
        struct_detail: 结构合理性详细说明。
        compliance_score: 规范合规性维度得分（未加权原始分）。
        compliance_detail: 规范合规性详细说明。
        total: 最终加权总分。
    """
    # 截断长 SQL 以便日志可读
    gen_preview = (
        generated_sql[:150] + ("…" if len(generated_sql) > 150 else "")
        if generated_sql
        else "(空)"
    )
    gold_preview = (
        gold_sql[:150] + ("…" if len(gold_sql) > 150 else "")
        if gold_sql
        else "(空)"
    )

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("[reward_sql] 奖励明细")
    lines.append(f"  db_id: {db_id}")
    lines.append(f"  生成SQL: {gen_preview}")
    lines.append(f"  GoldSQL: {gold_preview}")
    lines.append("-" * 40)
    lines.append(
        f"  ① 执行正确性 (权重{_WEIGHT_EXECUTION}): "
        f"原始={exec_score:.2f}  加权={exec_score * _WEIGHT_EXECUTION:.3f}"
    )
    lines.append(f"     {exec_detail}")
    lines.append(
        f"  ② 结构合理性 (权重{_WEIGHT_STRUCTURE}): "
        f"原始={struct_score:.2f}  加权={struct_score * _WEIGHT_STRUCTURE:.3f}"
    )
    lines.append(f"     {struct_detail}")
    lines.append(
        f"  ③ 规范合规性 (权重{_WEIGHT_COMPLIANCE}): "
        f"原始={compliance_score:.2f}  加权={compliance_score * _WEIGHT_COMPLIANCE:.3f}"
    )
    lines.append(f"     {compliance_detail}")
    lines.append("-" * 40)
    lines.append(f"  ★ 总分: {total:.4f}")
    lines.append("=" * 60)
    print("\n".join(lines))


# ============================================================
# 核心奖励函数
# ============================================================
def compute_reward(
    generated_sql: str,
    gold_sql: str,
    db_id: str,
    verbose: bool = True,
) -> float:
    """计算单条 SQL 的三维加权奖励分数。

    三维评分：
        1. 执行正确性（权重 0.6）：
           - 用 db_utils.execute_sql 分别执行生成 SQL 和 gold_sql
           - 语法错 / 不可执行 → 0 分
           - 可执行且结果集合一致（忽略列/行顺序）→ 满分（1.0）
           - 可执行但结果不一致 → 0.2 分（部分给分，鼓励生成可执行 SQL）
        2. 结构合理性（权重 0.2）：
           - 提取 SQL 的结构特征关键词（JOIN/GROUP BY/HAVING/子查询等）
           - 与 gold_sql 的结构特征做命中率比较
           - 得分 = 命中特征数 / gold_sql 特征数（gold 无特征时默认满分）
        3. 规范合规性（权重 0.2）：
           - 无 SELECT * → 通过（含 SELECT * 的扣至 0.5）
           - 无 DROP/DELETE/UPDATE/INSERT 等高危操作 → 通过
           - 命中高危关键字 → 总分直接归零

    最终 reward = 0.6*执行 + 0.2*结构 + 0.2*规范

    Args:
        generated_sql: 模型生成的 SQL 语句。
        gold_sql: 参考答案的 SQL 语句。
        db_id: 数据库标识符（对应 Spider 数据集中的数据库名，如 "department_store"）。
        verbose: 是否打印明细日志，默认 True。

    Returns:
        float: 奖励分数，范围 [0, 1]。
    """
    # ---- 空值保护 ----
    if not generated_sql or not generated_sql.strip():
        if verbose:
            _print_reward_detail(
                generated_sql="",
                gold_sql=gold_sql or "",
                db_id=db_id,
                exec_score=0.0,
                exec_detail="生成 SQL 为空",
                struct_score=0.0,
                struct_detail="生成 SQL 为空，无法分析结构",
                compliance_score=0.0,
                compliance_detail="生成 SQL 为空",
                total=0.0,
            )
        return 0.0

    sql = generated_sql.strip()

    # ---- 高危关键字直接归零（最高优先级） ----
    # 使用词边界匹配提取所有单词，与禁止关键字集取交集
    words_in_sql = set(re.findall(r"\b(\w+)\b", sql.upper()))
    danger_hits = _FORBIDDEN_KEYWORDS & words_in_sql
    if danger_hits:
        if verbose:
            _print_reward_detail(
                generated_sql=sql,
                gold_sql=gold_sql or "",
                db_id=db_id,
                exec_score=0.0,
                exec_detail="未执行（高危关键字命中，直接归零）",
                struct_score=0.0,
                struct_detail="未评估（高危关键字命中，直接归零）",
                compliance_score=0.0,
                compliance_detail=f"命中高危关键字: {sorted(danger_hits)} → 总分归零",
                total=0.0,
            )
        return 0.0

    # ================================================================
    # 维度 1: 执行正确性（权重 0.6）
    # ================================================================
    exec_score: float
    exec_detail: str

    gen_result = execute_sql(db_id=db_id, sql=sql)
    gold_result = execute_sql(db_id=db_id, sql=gold_sql) if gold_sql else None

    if not gen_result["success"]:
        # 生成 SQL 不可执行（语法错 / 表名错 / 超时等）
        exec_score = _EXEC_FAIL
        exec_detail = f"生成 SQL 执行失败: {gen_result.get('error', '未知错误')}"
    elif gold_result is None or not gold_result.get("success"):
        # gold_sql 不可用，无法对比结果；生成 SQL 可执行 → 给部分分
        gen_rows = len(gen_result.get("rows", []))
        exec_score = _EXEC_PARTIAL
        exec_detail = (
            f"生成 SQL 可执行（{gen_rows} 行），"
            f"但 gold_sql 执行失败，无法对比 → 部分给分 ({_EXEC_PARTIAL})"
        )
    else:
        # 两边都可执行 → 比较结果集合
        is_match, match_detail = _compare_sql_results(gen_result, gold_result)
        if is_match:
            exec_score = _EXEC_FULL
            gen_rows = len(gen_result.get("rows", []))
            exec_detail = f"结果完全一致（{gen_rows} 行）"
        else:
            exec_score = _EXEC_PARTIAL
            gen_rows = len(gen_result.get("rows", []))
            gold_rows = len(gold_result.get("rows", []))
            exec_detail = (
                f"可执行但结果不一致（生成={gen_rows}行, Gold={gold_rows}行）: {match_detail}"
            )

    # ================================================================
    # 维度 2: 结构合理性（权重 0.2）
    # ================================================================
    struct_score: float
    struct_detail: str

    gen_features = _extract_structure_features(sql)
    gold_features = _extract_structure_features(gold_sql) if gold_sql else set()

    if not gold_features:
        # gold_sql 无特殊结构（简单 SELECT），生成 SQL 也无特殊结构 → 满分
        if not gen_features:
            struct_score = 1.0
            struct_detail = "gold_sql 为简单查询，生成 SQL 也为简单查询 → 满分"
        else:
            # 生成 SQL 多了不必要的结构（过度复杂）
            struct_score = 0.5
            struct_detail = (
                f"gold_sql 为简单查询（无结构特征），"
                f"但生成 SQL 多出结构: {sorted(gen_features)} → 扣分至 0.5"
            )
    else:
        matched = gen_features & gold_features
        hit_ratio = len(matched) / len(gold_features)
        struct_score = hit_ratio
        only_gen = gen_features - gold_features
        only_gold = gold_features - gen_features
        parts: list[str] = [
            f"命中率={len(matched)}/{len(gold_features)}={hit_ratio:.2f}"
        ]
        if matched:
            parts.append(f"命中: {sorted(matched)}")
        if only_gold:
            parts.append(f"缺失: {sorted(only_gold)}")
        if only_gen:
            parts.append(f"多余: {sorted(only_gen)}")
        struct_detail = " | ".join(parts)

    # ================================================================
    # 维度 3: 规范合规性（权重 0.2）
    # ================================================================
    compliance_score: float
    compliance_detail: str

    compliant, reason = _check_compliance(sql)
    if compliant:
        compliance_score = 1.0
        compliance_detail = "通过: 无 SELECT *，无高危写操作"
    else:
        # SELECT * 扣分但不归零（给 0.5），高危关键字已在前面归零
        if "SELECT *" in reason:
            compliance_score = 0.5
            compliance_detail = f"部分通过: {reason}（扣至 0.5）"
        else:
            compliance_score = 0.0
            compliance_detail = f"不通过: {reason}"

    # ---- 加权总分 ----
    total = (
        _WEIGHT_EXECUTION * exec_score
        + _WEIGHT_STRUCTURE * struct_score
        + _WEIGHT_COMPLIANCE * compliance_score
    )
    # 确保分数在 [0, 1] 范围内（浮点精度保护）
    total = max(0.0, min(1.0, total))

    # ---- 打印明细日志 ----
    if verbose:
        _print_reward_detail(
            generated_sql=sql,
            gold_sql=gold_sql or "",
            db_id=db_id,
            exec_score=exec_score,
            exec_detail=exec_detail,
            struct_score=struct_score,
            struct_detail=struct_detail,
            compliance_score=compliance_score,
            compliance_detail=compliance_detail,
            total=total,
        )

    return total


# ============================================================
# 批量接口
# ============================================================
def batch_reward(
    samples: list[dict[str, Any]],
    verbose: bool = True,
) -> list[float]:
    """批量计算奖励分数，适配 LLaMA-Factory GRPO samples 格式。

    每个 sample 字典需包含以下字段（兼容多种键名）：
        - generated_sql 或 predict: 模型生成的 SQL 语句
        - gold_sql 或 output: 参考答案的 SQL 语句
        - db_id: 数据库标识符

    Args:
        samples: 样本字典列表。
        verbose: 是否打印日志，默认 True。
            - 单条样本：打印完整明细
            - 多条样本：每条打印单行摘要 + 汇总统计

    Returns:
        list[float]: 每个样本对应的奖励分数 [0, 1]，顺序与输入一致。

    Raises:
        ValueError: 样本缺少 db_id 字段时抛出。
    """
    scores: list[float] = []
    total_samples = len(samples)

    if verbose and total_samples > 0:
        print(f"\n[reward_sql] 批量评估开始，共 {total_samples} 条样本")

    for i, sample in enumerate(samples):
        # 兼容多种键名：优先取 generated_sql / gold_sql，回退到 predict / output
        generated_sql = str(sample.get("generated_sql") or sample.get("predict") or "")
        gold_sql = str(sample.get("gold_sql") or sample.get("output") or "")
        db_id = str(sample.get("db_id", ""))

        if not db_id:
            raise ValueError(f"第 {i} 条样本缺少 db_id 字段，请检查样本数据")

        # 单条样本打印完整明细，多条样本只打印单行摘要
        single_verbose = verbose and total_samples == 1
        score = compute_reward(
            generated_sql=generated_sql,
            gold_sql=gold_sql,
            db_id=db_id,
            verbose=single_verbose,
        )
        scores.append(score)

        if verbose and total_samples > 1:
            # 批量模式：每条打印一行摘要
            sql_preview = generated_sql[:60] + ("…" if len(generated_sql) > 60 else "")
            print(f"  [{i + 1}/{total_samples}] reward={score:.4f}  {sql_preview}")

    if verbose and total_samples > 0:
        avg = sum(scores) / total_samples
        min_score = min(scores)
        max_score = max(scores)
        # 统计分布
        high = sum(1 for s in scores if s >= 0.8)
        mid = sum(1 for s in scores if 0.4 <= s < 0.8)
        low = sum(1 for s in scores if s < 0.4)
        print(
            f"[reward_sql] 批量评估完成  "
            f"平均={avg:.4f}  最高={max_score:.4f}  最低={min_score:.4f}  "
            f"分布: ≥0.8={high}  0.4~0.8={mid}  <0.4={low}"
        )

    return scores


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    import os

    # 强制输出为 UTF-8，避免 Windows GBK 终端编码报错
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("=" * 60)
    print("  reward_sql self-test")
    print("=" * 60)

    test_db_id = "department_store"

    # ---- 测试用例 1: SQL 完全一致 → 应得满分 ----
    print("\n>>> Test 1: SQL exact match")
    gold1 = "SELECT COUNT(*) FROM Customers"
    gen1 = "SELECT COUNT(*) FROM Customers"
    score1 = compute_reward(gen1, gold1, test_db_id)
    expected1 = (
        _WEIGHT_EXECUTION * _EXEC_FULL
        + _WEIGHT_STRUCTURE * 1.0
        + _WEIGHT_COMPLIANCE * 1.0
    )
    status1 = "PASS" if abs(score1 - expected1) < 0.01 else "FAIL"
    print(f"  expected=~{expected1:.4f}  actual={score1:.4f}  [{status1}]")

    # ---- 测试用例 2: SQL 等价但列序不同（使用正确的列名） ----
    print("\n>>> Test 2: SQL equivalent, column order swapped")
    gold2 = "SELECT customer_id, customer_name FROM Customers LIMIT 3"
    gen2 = "SELECT customer_name, customer_id FROM Customers LIMIT 3"
    score2 = compute_reward(gen2, gold2, test_db_id)
    status2 = "PASS" if score2 > 0.9 else "FAIL"
    print(f"  expected=~1.0000  actual={score2:.4f}  [{status2}]")

    # ---- 测试用例 3: 使用 SELECT * → 规范合规性扣分 ----
    print("\n>>> Test 3: SELECT * penalty")
    gold3 = "SELECT customer_id, customer_name FROM Customers LIMIT 3"
    gen3 = "SELECT * FROM Customers LIMIT 3"
    score3 = compute_reward(gen3, gold3, test_db_id)
    status3 = "PASS" if score3 < 1.0 else "FAIL"
    print(f"  expected=<1.0 (compliance penalty)  actual={score3:.4f}  [{status3}]")

    # ---- 测试用例 4: 包含 DELETE 高危关键字 → 直接归零 ----
    print("\n>>> Test 4: DELETE keyword -> zero")
    gold4 = "SELECT customer_id FROM Customers"
    gen4 = "DELETE FROM Customers WHERE customer_id = 1"
    score4 = compute_reward(gen4, gold4, test_db_id)
    status4 = "PASS" if score4 == 0.0 else "FAIL"
    print(f"  expected=0.0000  actual={score4:.4f}  [{status4}]")

    # ---- 测试用例 5: 语法错误 → 执行分归零 ----
    print("\n>>> Test 5: syntax error -> exec score zero")
    gold5 = "SELECT COUNT(*) FROM Customers"
    gen5 = "SELEC COUNT(*) FROM Customers"  # SELECT 拼写错误
    score5 = compute_reward(gen5, gold5, test_db_id)
    expected5 = 0.0 * _WEIGHT_EXECUTION + 1.0 * _WEIGHT_STRUCTURE + 1.0 * _WEIGHT_COMPLIANCE
    status5 = "PASS" if abs(score5 - expected5) < 0.01 else "FAIL"
    print(f"  expected=~{expected5:.4f}  actual={score5:.4f}  [{status5}]")

    # ---- 测试用例 6: 空 SQL → 直接零分 ----
    print("\n>>> Test 6: empty SQL -> zero")
    gold6 = "SELECT COUNT(*) FROM Customers"
    score6 = compute_reward("", gold6, test_db_id)
    status6 = "PASS" if score6 == 0.0 else "FAIL"
    print(f"  expected=0.0000  actual={score6:.4f}  [{status6}]")

    # ---- 测试用例 7: 批量接口 ----
    print("\n>>> Test 7: batch_reward")
    samples = [
        {
            "generated_sql": "SELECT COUNT(*) FROM Customers",
            "gold_sql": "SELECT COUNT(*) FROM Customers",
            "db_id": test_db_id,
        },
        {
            "generated_sql": "SELECT * FROM Customers LIMIT 3",
            "gold_sql": "SELECT customer_id, customer_name FROM Customers LIMIT 3",
            "db_id": test_db_id,
        },
        {
            "generated_sql": "DELETE FROM Customers",
            "gold_sql": "SELECT COUNT(*) FROM Customers",
            "db_id": test_db_id,
        },
    ]
    scores = batch_reward(samples)
    print(f"  batch scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  expected: [~1.0, <1.0, 0.0]")

    # ---- 测试用例 8: 兼容 predict/output 键名 ----
    print("\n>>> Test 8: compatible predict/output keys")
    samples_alt = [
        {
            "predict": "SELECT COUNT(*) FROM Customers",
            "output": "SELECT COUNT(*) FROM Customers",
            "db_id": test_db_id,
        },
    ]
    scores_alt = batch_reward(samples_alt)
    print(f"  scores: {[f'{s:.4f}' for s in scores_alt]}")

    print("\n" + "=" * 60)
    print("  reward_sql self-test complete")
    print("=" * 60)
