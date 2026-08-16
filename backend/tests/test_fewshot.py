"""
测试 app.fewshot — 检索式 few-shot 模块。
覆盖: 分词（中文/英文/数字）、示例检索（db 过滤 + 排序）、文本格式化。
"""

from app.fewshot import _tokenize, format_examples, retrieve_examples


class TestTokenize:
    """_tokenize 分词功能测试。"""

    def test_tokenizes_chinese(self) -> None:
        """中文按字切分，用于关键词相似度匹配。"""
        tokens = _tokenize("产品类型")
        assert tokens == {"产", "品", "类", "型"}

    def test_tokenizes_english_and_numbers(self) -> None:
        """英文单词与数字作为独立 token。"""
        tokens = _tokenize("products 700")
        assert "products" in tokens
        assert "700" in tokens

    def test_lowercases_english(self) -> None:
        """英文统一转小写。"""
        tokens = _tokenize("Products")
        assert "products" in tokens


class TestRetrieveExamples:
    """retrieve_examples 检索功能测试。"""

    def test_filters_by_db(self) -> None:
        """不存在的 db_id 无命中，返回空列表。"""
        result = retrieve_examples("产品类型", "nonexistent_db_xyz")
        assert result == []

    def test_returns_ranked_examples(self) -> None:
        """按相似度返回 top-k，且每条含 question 与 sql 字段。"""
        result = retrieve_examples("哪些产品的平均价格超过700？", "department_store", k=3)
        assert 0 < len(result) <= 3
        for item in result:
            assert "question" in item
            assert "sql" in item


class TestFormatExamples:
    """format_examples 格式化功能测试。"""

    def test_empty_returns_empty(self) -> None:
        """空列表返回空字符串。"""
        assert format_examples([]) == ""

    def test_formats_markdown(self) -> None:
        """非空列表渲染为含 SQL 代码块的 markdown。"""
        examples = [{"question": "测试问题", "sql": "SELECT 1;"}]
        text = format_examples(examples)
        assert "```sql" in text
        assert "SELECT 1" in text
