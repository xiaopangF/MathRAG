"""
MathRAG LLM 生成器
功能：调用 DeepSeek API，根据检索到的上下文生成答案
兼容本地运行（.env）和 Streamlit Cloud（st.secrets）
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple

# 尝试导入 streamlit（云端会有，本地可能没有）
try:
    import streamlit as st
except ImportError:
    st = None

from openai import OpenAI


class LLMGenerator:
    """大模型生成器"""

    def __init__(self):
        self.api_key = None
        self.base_url = "https://api.deepseek.com/v1"

        # ---------- 方式1：从 Streamlit Cloud Secrets 读取 ----------
        if st is not None:
            try:
                self.api_key = st.secrets.get("DEEPSEEK_API_KEY")
                self.base_url = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
                if self.api_key:
                    print("✅ 从 Streamlit Secrets 读取 API Key 成功")
            except Exception:
                pass

        # ---------- 方式2：从 .env 文件读取（本地运行） ----------
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                env_path = Path(__file__).parent.parent.parent / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                    self.api_key = os.getenv("DEEPSEEK_API_KEY")
                    self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
                    if self.api_key:
                        print(f"✅ 从 .env 文件读取 API Key 成功: {env_path}")
            except Exception as e:
                print(f"⚠️ 加载 .env 文件失败: {e}")

        # ---------- 检查是否获取到 API Key ----------
        if not self.api_key:
            raise ValueError(
                "❌ 未找到 DEEPSEEK_API_KEY！\n"
                "   - 本地运行：请在项目根目录创建 .env 文件，并写入 DEEPSEEK_API_KEY=sk-xxx\n"
                "   - 云端部署：请在 Streamlit Cloud 的 Secrets 中配置 DEEPSEEK_API_KEY"
            )

        # ---------- 初始化 OpenAI 客户端 ----------
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # 高等数学专用 System Prompt
        self.system_prompt = """你是一位高等数学助教。请根据提供的教材内容，准确回答学生的问题。

规则：
1. 只使用提供的上下文来回答问题，不要编造教材中没有的内容。
2. 如果上下文中没有相关信息，请直接说"根据当前教材内容，未找到相关解释"。
3. 回答要清晰、有条理，用中文输出。
4. 对于数学公式，可以用文字描述，如果上下文中有 LaTeX 格式，可以保留。
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

        # 构造用户消息
        user_message = f"""学生问题：{query}

以下是从教材中检索到的相关内容：
{context_text}

请根据以上教材内容，回答学生的问题。"""

        # 调用 API
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