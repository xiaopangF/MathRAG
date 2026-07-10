"""
MathRAG LLM 生成器
功能：调用 DeepSeek API，根据检索到的上下文生成答案
"""
import os
from pathlib import Path
from typing import Any
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


class LLMAuthenticationError(RuntimeError):
    """Raised when the configured LLM API key is rejected by the provider."""


class LLMQuotaError(RuntimeError):
    """Raised when the provider account has no usable quota or balance."""


class LLMRateLimitError(RuntimeError):
    """Raised when the provider rate limit is exceeded."""


class LLMConnectionError(RuntimeError):
    """Raised when the provider cannot be reached."""


class LLMGenerationError(RuntimeError):
    """Raised when the LLM provider request fails for a non-auth reason."""


class LLMGenerator:
    """大模型生成器"""

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
        except Exception:
            pass

        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = (
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
            or "https://api.deepseek.com/v1"
        )
        self.timeout_seconds = self._resolve_timeout(timeout_seconds)
        self.max_retries = self._resolve_max_retries(max_retries)

        if not self.api_key:
            raise ValueError(
                "❌ 未找到 DEEPSEEK_API_KEY！\n"
                "请在运行前设置环境变量或在 .env 文件中配置。"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        self.system_prompt = r"""你是一位高等数学助教。请根据提供的教材内容，准确回答学生的问题。

规则：
1. 只使用提供的上下文来回答问题，不要编造教材中没有的内容。
2. 如果上下文中没有相关信息，请直接说"根据当前教材内容，未找到相关解释"。
3. 回答要清晰、有条理，用中文输出。
4. 当答案中的结论来自某个参考片段时，请在句末用 [1]、[2] 这样的编号标注来源。
5. 【公式渲染要求】对于数学公式，请严格使用 Markdown + LaTeX 格式：
   - 块级公式（独立成行）：必须用 $$ ... $$ 包裹，例如：
     $$ \int_0^1 x^2 dx = \frac{1}{3} $$
   - 内联公式（在行内）：必须用 $ ... $ 包裹，例如：
     导数 $f'(x)$ 表示函数在 $x$ 点的变化率。
   - 不要用普通括号包裹公式，例如不要写 ( f'(x) )，应写成 $f'(x)$。
"""

    @staticmethod
    def _resolve_timeout(value: float | None) -> float:
        if value is None:
            try:
                value = float(os.getenv("MATHRAG_LLM_TIMEOUT_SECONDS", "30"))
            except ValueError as exc:
                raise ValueError("MATHRAG_LLM_TIMEOUT_SECONDS 必须是数字") from exc
        if not 1 <= value <= 300:
            raise ValueError("LLM timeout 必须在 1 到 300 秒之间")
        return value

    @staticmethod
    def _resolve_max_retries(value: int | None) -> int:
        if value is None:
            try:
                value = int(os.getenv("MATHRAG_LLM_MAX_RETRIES", "2"))
            except ValueError as exc:
                raise ValueError("MATHRAG_LLM_MAX_RETRIES 必须是整数") from exc
        if not 0 <= value <= 10:
            raise ValueError("LLM max_retries 必须在 0 到 10 之间")
        return value

    @staticmethod
    def _format_page_range(page_start: Any, page_end: Any) -> str:
        """Format a source page range for prompt metadata."""
        if page_start in (None, ""):
            return ""
        if page_end in (None, "") or page_end == page_start:
            return str(page_start)
        return f"{page_start}-{page_end}"

    @staticmethod
    def _provider_message(error: Exception) -> str:
        response = getattr(error, "response", None)
        if response is None:
            return str(error)
        try:
            data = response.json()
        except Exception:
            return str(error)
        provider_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(provider_error, dict):
            return str(provider_error.get("message") or provider_error.get("code") or error)
        return str(error)

    @classmethod
    def _raise_for_status_error(cls, error: APIStatusError) -> None:
        status_code = getattr(error, "status_code", None)
        provider_message = cls._provider_message(error).lower()

        if status_code == 401:
            raise LLMAuthenticationError(
                "DeepSeek API Key 无效或已失效，请重新复制控制台里的有效 Key。"
            ) from error
        if status_code == 402 or "insufficient" in provider_message or "balance" in provider_message:
            raise LLMQuotaError(
                "DeepSeek 账户余额或额度不足，请检查控制台余额、套餐或充值状态。"
            ) from error
        if status_code == 403:
            raise LLMGenerationError(
                "DeepSeek 请求被拒绝，请检查账号权限、模型权限或 API Key 使用范围。"
            ) from error
        if status_code == 429:
            raise LLMRateLimitError(
                "DeepSeek 请求过于频繁，已触发限流，请稍后再试。"
            ) from error
        if status_code and status_code >= 500:
            raise LLMGenerationError(
                "DeepSeek 服务暂时不可用，请稍后重试。"
            ) from error

        raise LLMGenerationError(
            f"DeepSeek API 调用失败（HTTP {status_code or 'unknown'}），请检查请求参数和模型配置。"
        ) from error

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
            content = response.choices[0].message.content
            if not content:
                raise LLMGenerationError("DeepSeek 返回了空答案，请稍后重试。")
            return content
        except AuthenticationError as e:
            raise LLMAuthenticationError(
                "DeepSeek API Key 无效或已失效，请重新复制控制台里的有效 Key。"
            ) from e
        except PermissionDeniedError as e:
            raise LLMGenerationError(
                "DeepSeek 请求被拒绝，请检查账号权限、模型权限或 API Key 使用范围。"
            ) from e
        except RateLimitError as e:
            raise LLMRateLimitError(
                "DeepSeek 请求过于频繁，已触发限流，请稍后再试。"
            ) from e
        except (APITimeoutError, APIConnectionError) as e:
            raise LLMConnectionError(
                "无法连接 DeepSeek 服务，请检查网络、代理或 DEEPSEEK_BASE_URL。"
            ) from e
        except BadRequestError as e:
            raise LLMGenerationError(
                "DeepSeek 请求参数不正确，请检查模型名称、上下文长度或 base URL。"
            ) from e
        except APIStatusError as e:
            self._raise_for_status_error(e)
        except Exception as e:
            raise LLMGenerationError(f"生成答案时出错：{str(e)}") from e


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
