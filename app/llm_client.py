"""
LLM 客户端模块。
统一封装 LLM 调用，支持结构化 JSON 输出，预留本地模型切换。
使用 langchain_openai.ChatOpenAI，从 .env 读取 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL。
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持任意目录运行本文件
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_openai import ChatOpenAI

from app.log_utils import get_logger

# ============================================================
# 加载 .env 文件中的环境变量（使用 python-dotenv）
# ============================================================
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path, override=False)
except ImportError:
    # 若 python-dotenv 未安装，回退到手动解析
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.is_file():
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _value = _line.partition("=")
                os.environ.setdefault(_key.strip(), _value.strip())

# ============================================================
# 本地模型开关 — 后续 GRPO 微调模型无缝替换
# ============================================================
USE_LOCAL_MODEL = (
    os.environ.get("USE_LOCAL_MODEL", "False").lower() in ("1", "true", "yes")
)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """返回配置好的 ChatOpenAI 实例。

    从环境变量读取 API 配置：
        - LLM_API_KEY: API 密钥
        - LLM_BASE_URL: API 基础 URL（兼容 OpenAI / DeepSeek / 本地服务）
        - LLM_MODEL: 模型名称
        - LLM_REQUEST_TIMEOUT: LLM 单次调用超时秒数，默认 60s

    当 USE_LOCAL_MODEL=True 时打印 TODO 提示，但仍使用相同接口，
    便于后续 GRPO 微调模型无缝替换。

    Args:
        temperature: 采样温度，默认 0.0（确定性输出，适合 SQL 生成 / JSON 解析）。

    Returns:
        ChatOpenAI: 配置完成的 LLM 客户端实例。
    """
    if USE_LOCAL_MODEL:
        # TODO: 本地模型切换 — 后续替换为本地 vLLM/Ollama 部署的 GRPO 微调模型
        print("TODO本地模型: 当前仍使用 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL 配置的远端接口")

    # LLM 单次调用超时（可通过环境变量 LLM_REQUEST_TIMEOUT 覆盖，默认 60s）
    llm_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))

    return ChatOpenAI(
        api_key=os.environ.get("LLM_API_KEY", "sk-placeholder"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        temperature=temperature,
        request_timeout=llm_timeout,
    )


# ============================================================
# JSON 提取工具
# ============================================================
def _extract_json(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON 对象。

    尝试顺序：
        1. 去掉 ```json / ``` 代码块包裹后 json.loads
        2. 直接 json.loads 整个文本
        3. 正则提取首个 {...} 或 {...} 块后 json.loads

    Args:
        text: LLM 返回的原始文本。

    Returns:
        dict: 解析出的 JSON 字典。

    Raises:
        ValueError: 无法提取有效 JSON 字典。
    """
    cleaned = text.strip()

    # 1. 去掉 Markdown 代码块包裹 ```json ... ```
    fenced_match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    # 2. 尝试直接解析整个文本
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. 正则提取首个 JSON 对象 {...}
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError(f"无法从 LLM 输出中提取有效 JSON 字典，原始文本前 500 字符: {text[:500]}")


# ============================================================
# 结构化 JSON 调用
# ============================================================
def call_json(
    prompt: str,
    schema_hint: str,
    trace_id: str,
    max_retry: int = 2,
) -> dict:
    """调用 LLM 并要求返回结构化 JSON。

    在 prompt 末尾追加强制 JSON 输出的指令与 schema_hint。
    JSON 解析失败时追加错误提示重试，最多 max_retry 次。
    全部失败抛 ValueError。

    Args:
        prompt: 原始提示词。
        schema_hint: 期望的 JSON 结构描述（如 '{"answer": "整数"}'），
                     会追加到 prompt 末尾。
        trace_id: 日志追踪 ID。
        max_retry: JSON 解析失败最大重试次数，默认 2。

    Returns:
        dict: LLM 返回的 JSON 字典。

    Raises:
        ValueError: 超过最大重试次数仍无法解析有效 JSON。
    """
    log = get_logger(trace_id)
    llm = get_llm(temperature=0.0)

    # 构造强制 JSON 输出后缀
    json_suffix = (
        f"\n\n【重要】请严格只输出一个 JSON 对象，"
        f"不要包含 Markdown 代码块、解释文字或任何非 JSON 内容。"
        f"\n期望的 JSON 结构: {schema_hint}"
    )
    full_prompt = prompt + json_suffix

    last_output = ""
    for attempt in range(1, max_retry + 2):  # 1 次初始调用 + max_retry 次重试
        t0 = time.perf_counter()

        # 构造本次完整 prompt（重试时追加错误提示）
        if attempt == 1:
            current_prompt = full_prompt
        else:
            current_prompt = (
                full_prompt
                + f"\n\n【错误】上次输出无法解析为 JSON，请严格只输出一个 JSON 对象，"
                + f"不要包含任何额外文字。（重试 {attempt - 1}/{max_retry}）"
            )

        # 调用 LLM（带 request_timeout，超时时捕获并记录日志后重试）
        try:
            response = llm.invoke(current_prompt)
        except Exception as llm_exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log.warning(
                f"call_json LLM 调用异常  耗时 {elapsed_ms:.0f}ms  "
                f"尝试={attempt}/{max_retry + 1}  "
                f"异常={type(llm_exc).__name__}: {llm_exc}"
            )
            if attempt < max_retry + 1:
                continue  # 还有机会，重试下一轮
            # 全部重试失败
            log.error(f"call_json 全部 {max_retry + 1} 次尝试均失败")
            raise ValueError(
                f"LLM 在 {max_retry + 1} 次尝试后仍无法返回有效 JSON。"
                f"最后一次异常: {type(llm_exc).__name__}: {llm_exc}"
            ) from llm_exc

        last_output = response.content if hasattr(response, "content") else str(response)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 尝试解析 JSON
        try:
            result = _extract_json(last_output)
            log.info(
                f"call_json 完成  耗时 {elapsed_ms:.0f}ms  "
                f"解析成功  keys={list(result.keys())}  尝试={attempt}"
            )
            return result
        except ValueError:
            log.warning(
                f"call_json JSON 解析失败  耗时 {elapsed_ms:.0f}ms  "
                f"尝试={attempt}/{max_retry + 1}"
            )

    # 全部重试失败
    log.error(f"call_json 全部 {max_retry + 1} 次尝试均无法解析 JSON")
    raise ValueError(
        f"LLM 在 {max_retry + 1} 次尝试后仍无法返回有效 JSON。"
        f"最后输出前 500 字符: {last_output[:500]}"
    )


# ============================================================
# 普通文本调用
# ============================================================
def call_text(prompt: str, trace_id: str) -> str:
    """普通文本调用 LLM，用于 SQL 生成等场景。

    Args:
        prompt: 提示词文本。
        trace_id: 日志追踪 ID。

    Returns:
        str: LLM 返回的文本内容。
    """
    log = get_logger(trace_id)
    llm = get_llm(temperature=0.0)

    t0 = time.perf_counter()
    try:
        response = llm.invoke(prompt)
    except Exception as llm_exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.error(
            f"call_text LLM 调用异常  耗时 {elapsed_ms:.0f}ms  "
            f"异常={type(llm_exc).__name__}: {llm_exc}"
        )
        raise

    text = response.content if hasattr(response, "content") else str(response)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    log.info(
        f"call_text 完成  耗时 {elapsed_ms:.0f}ms  "
        f"输出长度={len(text)}  chars"
    )
    return text


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    from log_utils import new_trace_id

    tid = new_trace_id()
    log = get_logger(tid)
    log.info("=== llm_client 自测开始 ===")

    # ---------- 自测 1: get_llm ----------
    llm = get_llm()
    model_name = getattr(llm, "model_name", "?")
    base_url = getattr(llm, "openai_api_base", "?")
    log.info(f"get_llm 返回实例: model={model_name}, base_url={base_url}")

    # ---------- 自测 2: call_json — 让模型返回 {"answer": 42} ----------
    log.info('自测 call_json: 请求 LLM 返回 {"answer": 42}')
    try:
        result = call_json(
            prompt="请返回一个 JSON 对象，包含键 answer，值为数字 42。",
            schema_hint='{"answer": 42}',
            trace_id=tid,
            max_retry=2,
        )
        assert result.get("answer") == 42, (
            f"期望 answer=42，实际 answer={result.get('answer')}"
        )
        log.info(f"[PASS] call_json 自测通过: result={json.dumps(result, ensure_ascii=False)}")
    except Exception as e:
        log.error(f"[FAIL] call_json 自测失败: {e}")

    # ---------- 自测 3: call_text ----------
    log.info("自测 call_text: 简单文本调用")
    try:
        text = call_text(
            prompt="用一句话介绍 SQL 是什么。",
            trace_id=tid,
        )
        log.info(f"[PASS] call_text 自测通过: 输出长度={len(text)}, 前80字符='{text[:80]}'")
    except Exception as e:
        log.error(f"[FAIL] call_text 自测失败: {e}")

    log.info("=== llm_client 自测结束 ===")
