"""
MathRAG Web 界面
基于 Streamlit 的高等数学知识库问答系统
"""
import sys
from pathlib import Path
import os

# 把项目根目录加到 Python 路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import tempfile
import streamlit as st
from collections.abc import Iterable

# ============== 页面设置 ==============
st.set_page_config(
    page_title="MathRAG - 高数知识库问答",
    page_icon="📐",
    layout="wide"
)


# ============== 读取 API Key（从 .env、环境变量或页面临时输入） ==============

def get_deepseek_key():
    """获取 DeepSeek API Key，不在缺失时阻断页面渲染。"""
    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            key = os.getenv("DEEPSEEK_API_KEY")
            if key:
                return key
    except Exception as e:
        st.error(f"❌ 读取 .env 失败: {e}")

    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key

    return None


DEEPSEEK_API_KEY = get_deepseek_key()
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY


# ============== 导入项目模块 ==============
from src.loader.pdf_loader import PDFLoader
from src.splitter.structural_splitter import smart_split_by_titles, save_chunks_to_files
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


def has_local_knowledge_base():
    """检查本地是否已有可直接使用的知识库。"""
    index_path = project_root / "data/faiss_index/index.faiss"
    meta_path = project_root / "data/faiss_index/chunks_meta.jsonl"
    return index_path.exists(), meta_path.exists()


def show_model_loading_help(error):
    """展示模型加载失败时的可操作提示。"""
    st.session_state.model_load_failed = True
    st.error(f"❌ 失败: {error}")
    st.info(
        "这是 embedding/reranker 模型加载失败，不是 PDF 本身的问题。"
        "已有本地知识库只能跳过 PDF 上传和索引构建；"
        "但问答仍需要加载 embedding 模型来编码用户问题，并需要 reranker 模型重排结果。"
        "请确保能访问 HuggingFace/镜像站，或在 `.env` 中把 "
        "`MATHRAG_EMBEDDING_MODEL` 和 `MATHRAG_RERANKER_MODEL` 配成本地模型目录。"
    )


# ============== 侧边栏 ==============
with st.sidebar:
    st.title("📐 MathRAG")
    st.caption("基于双阶段检索的高等数学知识库问答系统")
    st.divider()

    st.subheader("📊 知识库状态")
    index_exists, meta_exists = has_local_knowledge_base()
    knowledge_ready = index_exists and meta_exists
    chunks_dir = project_root / "data/chunks"
    txt_files = list(chunks_dir.rglob("*.txt")) if chunks_dir.exists() else []
    api_ready = bool(os.environ.get("DEEPSEEK_API_KEY"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 知识块", len(txt_files))
    col2.metric("🔍 索引", "✅" if knowledge_ready else "❌")
    col3.metric("🔑 API", "✅" if api_ready else "❌")
    if st.session_state.get("retriever_initialized"):
        model_status = "✅"
    elif st.session_state.get("model_load_failed"):
        model_status = "❌"
    elif knowledge_ready and api_ready:
        model_status = "待加载"
    else:
        model_status = "❌"
    col4.metric("🧠 模型", model_status)

    if knowledge_ready:
        st.success("已检测到本地索引。")
    else:
        st.warning("未检测到本地知识库。可在下方上传 PDF 构建。")
        with st.expander("路径诊断", expanded=True):
            st.code(str(project_root))
            st.write("index.faiss:", (project_root / "data/faiss_index/index.faiss").exists())
            st.write("chunks_meta.jsonl:", (project_root / "data/faiss_index/chunks_meta.jsonl").exists())

    if not api_ready:
        st.warning("未配置 DeepSeek API Key，无法初始化问答模型。")
        api_key_input = st.text_input("临时输入 DeepSeek API Key", type="password")
        if api_key_input:
            os.environ["DEEPSEEK_API_KEY"] = api_key_input
            st.session_state.pipeline = None
            st.session_state.retriever_initialized = False
            st.session_state.model_load_failed = False
            st.rerun()

    st.divider()

    with st.expander("📤 更新或替换教材（可选）", expanded=not knowledge_ready):
        uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"], help="已有知识库时无需上传；上传后可重建本地知识库。")

        if uploaded_file is not None:
            st.success(f"✅ 已选择: {uploaded_file.name}")
            if st.button("🚀 重建知识库", type="primary", use_container_width=True):
                with st.spinner("正在处理 PDF、切分文本并重建索引..."):
                    pdf_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            pdf_path = tmp_file.name

                        st.info("正在提取 PDF 文本...")
                        loader = PDFLoader(pdf_path)
                        full_text = loader.extract_full_text()
                        loader.close()

                        st.info("正在结构化切分教材...")
                        chunks = smart_split_by_titles(full_text)
                        save_chunks_to_files(chunks, "data/chunks", clear_existing=True)

                        st.info("正在构建向量索引...")
                        from src.retriever.vector_indexer import build_vector_index
                        build_vector_index()

                        st.session_state.pipeline = None
                        st.session_state.retriever_initialized = False
                        st.success("🎉 知识库重建完成！")
                        st.rerun()
                    except Exception as e:
                        if "模型加载失败" in str(e) or "huggingface" in str(e).lower() or "hf-mirror" in str(e).lower():
                            show_model_loading_help(e)
                        else:
                            st.error(f"❌ 失败: {e}")
                    finally:
                        if pdf_path and Path(pdf_path).exists():
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
if "model_load_failed" not in st.session_state:
    st.session_state.model_load_failed = False

# 加载系统
api_ready = bool(os.environ.get("DEEPSEEK_API_KEY"))
if index_exists and meta_exists and api_ready and not st.session_state.retriever_initialized:
    try:
        with st.spinner("🔄 加载系统..."):
            st.session_state.pipeline = MathRAGPipeline()
            st.session_state.retriever_initialized = True
            st.session_state.model_load_failed = False
            st.success("✅ 系统加载完成！")
    except Exception as e:
        if "huggingface" in str(e).lower() or "hf-mirror" in str(e).lower():
            show_model_loading_help(e)
        else:
            st.error(f"❌ 加载失败: {e}")

if not index_exists or not meta_exists:
    st.warning("⚠️ 未检测到本地知识库。请在左侧“更新或替换教材（可选）”中上传 PDF 构建一次。")
    st.stop()

if not api_ready:
    st.warning("⚠️ 已检测到本地索引，但还没有配置 DeepSeek API Key。请在左侧输入临时 Key，或在项目根目录创建 `.env`。")
    st.stop()

if st.session_state.model_load_failed:
    st.warning("⚠️ API Key 已配置，但 embedding/reranker 模型未加载成功。请检查网络或配置本地模型目录。")
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
