"""
MathRAG LLM 生成器
功能：调用 DeepSeek API，根据检索到的上下文生成答案
"""
import os
from pathlib import Path
from typing import List, Tuple
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
4. 【公式渲染要求】对于数学公式，请严格使用 LaTeX 格式：
   - 块级公式（独立成行）：用 $$ ... $$ 包裹，例如：
     $$ \int_0^1 x^2 dx = \frac{1}{3} $$
   - 内联公式（在行内）：用 \( ... \) 包裹，例如：
     导数 \( f'(x) \) 表示函数在 \( x \) 点的变化率。
"""

    def generate(self, query: str, contexts: List[Tuple[str, float]]) -> str:
        """
        基于检索到的上下文生成答案
        Args:
            query: 用户问题
            contexts: [(content, score), ...] 检索到的知识块列表
        Returns:
            str: 生成的答案
        """
        # 拼接上下文
        context_text = ""
        for i, (content, score) in enumerate(contexts, 1):
            context_text += f"\n【参考片段 {i}】\n{content}\n"

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