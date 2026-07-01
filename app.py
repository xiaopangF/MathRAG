"""
MathRAG Web 界面（稳定版，无自定义 JS）
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import os
import tempfile
import streamlit as st

st.set_page_config(page_title="MathRAG - 高数知识库问答", page_icon="📐", layout="wide")

# ---------- API Key 读取 ----------
def get_deepseek_key():
    try:
        if hasattr(st, 'secrets') and st.secrets:
            key = st.secrets.get("DEEPSEEK_API_KEY")
            if key:
                return key
    except:
        pass
    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            key = os.getenv("DEEPSEEK_API_KEY")
            if key:
                return key
    except:
        pass
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    st.error("❌ 未找到 DEEPSEEK_API_KEY，请检查配置。")
    return None

DEEPSEEK_API_KEY = get_deepseek_key()
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
else:
    st.stop()

# ---------- 导入模块 ----------
from src.loader.pdf_loader import PDFLoader
from src.splitter.structural_splitter import smart_split_by_titles, save_chunks_to_files
from src.retriever.retriever import MathRAGRetriever
from src.pipeline.qa_pipeline import MathRAGPipeline

# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("📐 MathRAG")
    st.caption("基于双阶段检索的高等数学知识库问答系统")
    st.divider()

    st.subheader("📊 系统状态")
    chunks_dir = Path("data/chunks")
    index_exists = Path("data/faiss_index").exists()
    meta_exists = Path("data/chunks_meta.pkl").exists()
    chunks_exist = chunks_dir.exists() and len(list(chunks_dir.glob("*.txt"))) > 0

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 知识块", len(list(chunks_dir.glob("*.txt"))) if chunks_exist else 0)
    col2.metric("🔍 索引", "✅" if index_exists else "❌")
    col3.metric("🧠 模型", "✅" if index_exists else "❌")
    st.divider()

    st.subheader("📤 上传教材")
    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])
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

# ---------- 主区域 ----------
st.title("📚 高等数学知识库问答")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False

if index_exists and meta_exists and not st.session_state.retriever_initialized:
    try:
        with st.spinner("🔄 加载系统..."):
            st.session_state.pipeline = MathRAGPipeline()
            st.session_state.retriever_initialized = True
        st.success("✅ 系统加载完成！")
    except Exception as e:
        st.error(f"❌ 加载失败: {e}")

if not index_exists or not meta_exists:
    st.warning("⚠️ 请先上传 PDF 并构建知识库。")
    st.stop()

# ---------- 显示消息（不进行公式渲染，仅显示原文） ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # 不开启 unsafe_allow_html
        if message["role"] == "assistant" and "contexts" in message:
            with st.expander("📖 相关片段"):
                for i, (content, score) in enumerate(message["contexts"]):
                    st.caption(f"片段 {i+1} (得分 {score:.4f})")
                    st.text(content[:300] + "..." if len(content) > 300 else content)

# ---------- 输入 ----------
if prompt := st.chat_input("输入高数问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 生成中..."):
            try:
                result = st.session_state.pipeline.ask(prompt)
                answer = result["answer"]
                st.markdown(answer)  # 不渲染公式
                with st.expander("📖 相关片段"):
                    for i, (content, score) in enumerate(result["contexts"]):
                        st.caption(f"片段 {i+1} (得分 {score:.4f})")
                        st.text(content[:300] + "..." if len(content) > 300 else content)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "contexts": result["contexts"]
                })
            except Exception as e:
                st.error(f"❌ 出错: {e}")

if st.session_state.messages:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()