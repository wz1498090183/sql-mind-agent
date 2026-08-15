"""
测试 app.llm_client — JSON 提取工具和 .env 加载。
"""

import pytest

from app.llm_client import _extract_json, get_llm


class TestExtractJson:
    """JSON 提取函数测试。"""

    def test_bare_json_object(self) -> None:
        """纯 JSON 对象文本直接解析。"""
        result = _extract_json('{"answer": 42}')
        assert result == {"answer": 42}

    def test_json_inside_markdown_block(self) -> None:
        """```json ... ``` 包裹的 JSON。"""
        text = """```json
{"name": "test", "count": 10}
```"""
        result = _extract_json(text)
        assert result == {"name": "test", "count": 10}

    def test_json_inside_plain_block(self) -> None:
        """``` ... ``` 包裹的 JSON（无语言标注）。"""
        text = """```
{"key": "value"}
```"""
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_with_extra_text(self) -> None:
        """JSON 前后混杂额外文字 — 通过正则提取。"""
        text = "这是一段说明\n{\"status\": \"ok\"}\n还有尾注"
        result = _extract_json(text)
        assert result == {"status": "ok"}

    def test_invalid_json_raises(self) -> None:
        """无效 JSON → ValueError。"""
        with pytest.raises(ValueError, match="无法从 LLM 输出中提取"):
            _extract_json("这不是 JSON，只是一段文字。没有大括号")

    def test_nested_object(self) -> None:
        """嵌套 JSON 对象正确解析。"""
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = _extract_json(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_array_only_raises(self) -> None:
        """纯数组（非对象）→ ValueError（要求 dict）。"""
        with pytest.raises(ValueError):
            _extract_json("[1, 2, 3]")


class TestGetLlm:
    """LLM 客户端工厂测试。"""

    def test_returns_chat_openai(self) -> None:
        """get_llm 返回 ChatOpenAI 实例。"""
        llm = get_llm()
        assert llm is not None
        # ChatOpenAI 应该有 model_name 属性
        assert hasattr(llm, "model_name")
