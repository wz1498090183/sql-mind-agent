"""
检索式 few-shot 模块。
从本地示例池中按问题相似度（关键词 Jaccard 重叠）检索最相似的 SQL 示例，
动态注入 Prompt，替代静态硬编码 few-shot，提升同类问题的生成准确率。

用法:
    from app.fewshot import retrieve_examples, format_examples
    examples = retrieve_examples(question, db_id, k=3)
    text = format_examples(examples)
"""

import json
import re
from pathlib import Path
from typing import Any, cast

# 示例池文件路径（backend/data/fewshot_pool.json）
_POOL_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "fewshot_pool.json"

# 分词：ASCII 单词 + 单个 CJK 汉字（中文无空格分隔，按字切分做关键词匹配）
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> set[str]:
    """把问题文本切分为小写 token 集合（去重），用于相似度计算。"""
    return set(_TOKEN_RE.findall(text.lower()))


def _load_pool() -> list[dict]:
    """加载本地示例池。

    文件缺失或格式错误时返回空列表（调用方降级到静态 few-shot），
    不让示例池问题拖垮主流程。

    Returns:
        list[dict]: 示例列表 [{db_id, question, sql}, ...]。
    """
    try:
        with open(_POOL_PATH, encoding="utf-8") as f:
            data: Any = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return [cast(dict, item) for item in data if isinstance(item, dict)]
    return []


def retrieve_examples(question: str, db_id: str, k: int = 3) -> list[dict]:
    """检索与问题最相似的 k 个 SQL 示例。

    只从与当前 db_id 相同的示例中检索，按问题 token 的 Jaccard 相似度降序
    排序，返回 top-k。当前 db 无示例或问题为空时返回空列表。

    Args:
        question: 用户问题或子任务描述。
        db_id: 数据库标识符。
        k: 返回示例数。

    Returns:
        list[dict]: 按相似度降序的示例列表，无命中时为空列表。
    """
    candidates = [p for p in _load_pool() if p.get("db_id") == db_id]
    if not candidates:
        return []

    q_tokens = _tokenize(question)
    if not q_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for p in candidates:
        p_tokens = _tokenize(p.get("question", ""))
        if not p_tokens:
            continue
        overlap = len(q_tokens & p_tokens)
        union = len(q_tokens | p_tokens)
        jaccard = overlap / union if union else 0.0
        scored.append((jaccard, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def format_examples(examples: list[dict]) -> str:
    """把检索到的示例渲染成 few-shot 文本块，供注入 Prompt。

    Args:
        examples: retrieve_examples 返回的示例列表。

    Returns:
        str: few-shot 文本块，空列表时返回空字符串。
    """
    if not examples:
        return ""

    blocks = ["## 相似示例（检索自历史案例，表名以当前 Schema 为准）"]
    for i, ex in enumerate(examples, 1):
        sql = ex.get("sql", "").strip().rstrip(";")
        blocks.append(
            f"### 示例 {i}\n需求：{ex.get('question', '')}\n```sql\n{sql};\n```"
        )
    return "\n\n".join(blocks)
