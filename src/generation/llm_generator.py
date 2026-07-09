"""
MathRAG LLM 生成器
功能：调用 DeepSeek API，根据检索到的上下文生成答案
"""
import os
from pathlib import Path
from typing import Any
from openai import OpenAI


class LLMGenerator:
    """大模型生成器"""

    def __init__(self):
        # 直接从环境变量读取（app.py 已经设置好了）
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        if not self.api_key:
            # 如果环境变量没有，尝试从 .env 加载（兼容直接运行测试）
            try:
                from dotenv import load_dotenv
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    self.api_key = os.getenv("DEEPSEEK_API_KEY")
                    self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            except Exception:
                pass

        if not self.api_key:
            raise ValueError(
                "❌ 未找到 DEEPSEEK_API_KEY！\n"
                "请在运行前设置环境变量或在 .env 文件中配置。"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.system_prompt = r"""你是一位高等数学助教。请根据提供的教材内容，准确回答学生的问题。

规则：
1. 只使用提供的上下文来回答问题，不要编造教材中没有的内容。
2. 如果上下文中没有相关信息，请直接说"根据当前教材内容，未找到相关解释"。
3. 回答要清晰、有条理，用中文输出。
4. 当答案中的结论来自某个参考片段时，请在句末用 [1]、[2] 这样的编号标注来源。
5. 【公式渲染要求】对于数学公式，请严格使用 LaTeX 格式：
   - 块级公式（独立成行）：用 $$ ... $$ 包裹，例如：
     $$ \int_0^1 x^2 dx = \frac{1}{3} $$
   - 内联公式（在行内）：用 \( ... \) 包裹，例如：
     导数 \( f'(x) \) 表示函数在 \( x \) 点的变化率。
"""

    @staticmethod
    def _format_page_range(page_start: Any, page_end: Any) -> str:
        """Format a source page range for prompt metadata."""
        if page_start in (None, ""):
            return ""
        if page_end in (None, "") or page_end == page_start:
            return str(page_start)
        return f"{page_start}-{page_end}"

    @staticmethod
    def _normalize_context(context: Any, index: int) -> dict:
        """兼容旧 tuple 格式和新的结构化 context 格式。"""
        if isinstance(context, dict):
            return {
                "index": index,
                "content": context.get("content", ""),
                "score": context.get("score", 0.0),
                "title": context.get("title", ""),
                "chapter": context.get("chapter", ""),
                "section": context.get("section", ""),
                "chunk_type": context.get("chunk_type", ""),
                "source_file": context.get("source_file", ""),
                "page_start": context.get("page_start"),
                "page_end": context.get("page_end"),
            }

        if isinstance(context, (list, tuple)) and len(context) >= 2:
            return {
                "index": index,
                "content": context[0],
                "score": context[1],
                "title": "",
                "chapter": "",
                "section": "",
                "chunk_type": "",
                "source_file": "",
                "page_start": None,
                "page_end": None,
            }

        return {
            "index": index,
            "content": str(context),
            "score": 0.0,
            "title": "",
            "chapter": "",
            "section": "",
            "chunk_type": "",
            "source_file": "",
            "page_start": None,
            "page_end": None,
        }

    def generate(self, query: str, contexts: list[Any]) -> str:
        """
        基于检索到的上下文生成答案
        Args:
            query: 用户问题
            contexts: 检索到的知识块列表，支持 dict 或 (content, score)
        Returns:
            str: 生成的答案
        """
        normalized_contexts = [
            self._normalize_context(context, i)
            for i, context in enumerate(contexts, 1)
        ]

        context_text = ""
        for item in normalized_contexts:
            metadata = []
            if item["source_file"]:
                metadata.append(f"来源文件: {item['source_file']}")
            page_range = self._format_page_range(item["page_start"], item["page_end"])
            if page_range:
                metadata.append(f"页码: {page_range}")
            if item["chapter"]:
                metadata.append(f"章节: {item['chapter']}")
            if item["section"]:
                metadata.append(f"小节: {item['section']}")
            if item["title"]:
                metadata.append(f"标题: {item['title']}")
            if item["chunk_type"]:
                metadata.append(f"类型: {item['chunk_type']}")
            metadata.append(f"相关性: {item['score']:.4f}")
            metadata_text = "\n".join(metadata)
            context_text += (
                f"\n【参考片段 {item['index']}】\n"
                f"{metadata_text}\n"
                f"内容:\n{item['content']}\n"
            )

        user_message = f"""学生问题：{query}

以下是从教材中检索到的相关内容：
{context_text}

请根据以上教材内容，回答学生的问题。"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=2048,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ 生成答案时出错：{str(e)}"


# ---------- 测试 ----------
if __name__ == "__main__":
    print("🧪 测试 LLM 生成器...")
    generator = LLMGenerator()
    test_contexts = [
        ("导数是函数在某一点的变化率，表示函数在该点的瞬时变化速度。", 0.9),
        ("导数的几何意义是切线的斜率。", 0.8)
    ]
    answer = generator.generate("什么是导数？", test_contexts)
    print("\n" + "="*50)
    print("📝 生成的答案：")
    print("="*50)
    print(answer)
