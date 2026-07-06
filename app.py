"""
MathRAG Web 界面
基于 Streamlit 的高等数学知识库问答系统
"""
import sys
from pathlib import Path

# 把项目根目录加到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import tempfile
import streamlit as st
from collections.abc import Iterable

# ============== 页面设置 ==============
st.set_page_config(
    page_title="MathRAG - 高数知识库问答",
    page_icon="📐",
    layout="wide"
)


# ============== 读取 API Key（只从 .env 和环境变量） ==============

def get_deepseek_key():
    """获取 DeepSeek API Key，只从 .env 和环境变量读取"""
    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            key = os.getenv("DEEPSEEK_API_KEY")
            if key:
                st.success("✅ 从 .env 文件读取 API Key 成功")
                return key
        else:
            st.warning(f"⚠️ .env 文件不存在: {env_path}，尝试从环境变量读取...")
    except Exception as e:
        st.error(f"❌ 读取 .env 失败: {e}")

    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        st.success("✅ 从环境变量读取 API Key 成功")
        return key

    st.error("""
    ❌ 未找到 DEEPSEEK_API_KEY！
    请在项目根目录创建 `.env` 文件，写入：
    DEEPSEEK_API_KEY=sk-你的真实密钥
    """)
    return None


DEEPSEEK_API_KEY = get_deepseek_key()
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
else:
    st.stop()


# ============== 导入项目模块 ==============
from src.loader.pdf_loader import PDFLoader
from src.splitter.structural_splitter import smart_split_by_titles, save_chunks_to_files
from src.retriever.retriever import MathRAGRetriever
from src.pipeline.qa_pipeline import MathRAGPipeline


# ============== 辅助函数：安全获取内容和分数 ==============
def get_chunk_content_score(chunk):
    """统一处理各种格式，返回 (content, score)"""
    if isinstance(chunk, dict):
        return chunk.get("content", ""), chunk.get("score", 0.0)
    if hasattr(chunk, 'content') and hasattr(chunk, 'rerank_score'):
        return chunk.content, chunk.rerank_score
    elif isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
        return chunk[0], chunk[1]
    else:
        return str(chunk), 0.0


def ensure_iterable(obj):
    """确保对象是可迭代的列表，如果不是则包装成单元素列表"""
    if obj is None:
        return []
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return list(obj)
    return [obj]


def format_source_label(chunk, index):
    """格式化引用来源标题"""
    if not isinstance(chunk, dict):
        content, score = get_chunk_content_score(chunk)
        return f"片段 {index} (相关性: {score:.4f})"

    title = chunk.get("title") or "未知标题"
    chapter = chunk.get("chapter") or ""
    section = chunk.get("section") or ""
    chunk_type = chunk.get("chunk_type") or ""
    score = chunk.get("score", 0.0)

    parts = [f"片段 {index}"]
    if chapter:
        parts.append(chapter)
    if section:
        parts.append(section)
    if title:
        parts.append(title)
    if chunk_type:
        parts.append(chunk_type)

    return " / ".join(parts) + f" / 相关性 {score:.4f}"


# ============== 侧边栏 ==============
with st.sidebar:
    st.title("📐 MathRAG")
    st.caption("基于双阶段检索的高等数学知识库问答系统")
    st.divider()

    st.subheader("📊 系统状态")
    index_exists = Path("data/faiss_index/index.faiss").exists()
    meta_exists = Path("data/faiss_index/chunks_meta.jsonl").exists()
    txt_files = list(Path("data/chunks").rglob("*.txt")) if Path("data/chunks").exists() else []

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 知识块", len(txt_files))
    col2.metric("🔍 索引", "✅" if index_exists else "❌")
    col3.metric("🧠 模型", "✅" if index_exists else "❌")
    st.divider()

    st.subheader("📤 上传教材")
    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"], help="上传后点击下方按钮构建知识库")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        st.success(f"✅ 已上传: {uploaded_file.name}")
        if st.button("🚀 构建知识库", type="primary", use_container_width=True):
            with st.spinner("正在处理..."):
                try:
                    loader = PDFLoader(pdf_path)
                    full_text = loader.extract_full_text()
                    loader.close()
                    chunks = smart_split_by_titles(full_text)
                    save_chunks_to_files(chunks, "data/chunks")
                    from src.retriever.vector_indexer import build_vector_index
                    build_vector_index()
                    st.success("🎉 构建完成！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 失败: {e}")
        os.unlink(pdf_path)

    st.divider()
    st.caption("电子科技大学 · 人工智能专业 · 暑假项目")


# ============== 主区域 ==============
st.title("📚 高等数学知识库问答")

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False

# 加载系统
if index_exists and meta_exists and not st.session_state.retriever_initialized:
    try:
        with st.spinner("🔄 加载系统..."):
            st.session_state.pipeline = MathRAGPipeline()
            st.session_state.retriever_initialized = True
            st.success("✅ 系统加载完成！")
    except Exception as e:
        st.error(f"❌ 加载失败: {e}")

if not index_exists or not meta_exists:
    st.warning("⚠️ 未检测到知识库，请在左侧上传 PDF 并构建。")
    st.stop()

# ========== 显示历史消息（带安全检查） ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "contexts" in message:
            with st.expander("📖 查看检索到的相关片段"):
                contexts = ensure_iterable(message["contexts"])
                for i, chunk in enumerate(contexts):
                    content, score = get_chunk_content_score(chunk)
                    st.caption(format_source_label(chunk, i + 1))
                    st.text(content[:300] + "..." if len(content) > 300 else content)

# ========== 输入框 ==========
if prompt := st.chat_input("输入你的高数问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                if st.session_state.pipeline is None:
                    st.error("系统未初始化，请刷新页面")
                    st.stop()

                result = st.session_state.pipeline.ask(prompt)
                st.markdown(result["answer"])

                # 显示检索片段（安全迭代）
                with st.expander("📖 查看检索到的相关片段"):
                    contexts = ensure_iterable(result.get("contexts", []))
                    for i, chunk in enumerate(contexts):
                        content, score = get_chunk_content_score(chunk)
                        st.caption(format_source_label(chunk, i + 1))
                        st.text(content[:300] + "..." if len(content) > 300 else content)

                # 保存历史，保留检索来源 metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": ensure_iterable(result.get("contexts", []))
                })

            except Exception as e:
                st.error(f"❌ 出错了: {e}")

# ========== 清空对话 ==========
if st.session_state.messages:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()
