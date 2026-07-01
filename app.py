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

# ============== 页面设置 ==============
st.set_page_config(
    page_title="MathRAG - 高数知识库问答",
    page_icon="📐",
    layout="wide"
)

# ============== 加载 KaTeX（支持公式渲染） ==============
def load_katex():
    """加载 KaTeX 的 CSS 和 JS，并准备渲染函数"""
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
        """,
        unsafe_allow_html=True
    )

load_katex()

# ============== 读取 API Key ==============
def get_deepseek_key():
    try:
        if hasattr(st, 'secrets') and st.secrets:
            key = st.secrets.get("DEEPSEEK_API_KEY")
            if key:
                return key
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            key = os.getenv("DEEPSEEK_API_KEY")
            if key:
                return key
    except Exception:
        pass

    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key

    st.error("""
    ❌ 未找到 DEEPSEEK_API_KEY！
    请参考部署文档配置。
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

# ============== 侧边栏 ==============
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
    uploaded_file = st.file_uploader(
        "上传 PDF 文件（高等数学教材）",
        type=["pdf"],
        help="上传后点击下方按钮构建知识库"
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        st.success(f"✅ 已上传: {uploaded_file.name}")

        if st.button("🚀 构建知识库", type="primary", use_container_width=True):
            with st.spinner("正在处理PDF，请稍候..."):
                try:
                    loader = PDFLoader(pdf_path)
                    full_text = loader.extract_full_text()
                    loader.close()

                    chunks = smart_split_by_titles(full_text)
                    save_chunks_to_files(chunks, "data/chunks")

                    from src.retriever.vector_indexer import build_vector_index
                    build_vector_index()

                    st.success("🎉 知识库构建完成！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 构建失败: {e}")
        os.unlink(pdf_path)

    st.divider()
    st.caption("电子科技大学 · 人工智能专业 · 暑假项目")

# ============== 主区域 ==============
st.title("📚 高等数学知识库问答")

# 初始化 session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False

# 加载系统
if index_exists and meta_exists and not st.session_state.retriever_initialized:
    try:
        with st.spinner("🔄 正在加载系统..."):
            st.session_state.pipeline = MathRAGPipeline()
            st.session_state.retriever_initialized = True
        st.success("✅ 系统加载完成！可以开始提问了。")
    except Exception as e:
        st.error(f"❌ 系统加载失败: {e}")

if not index_exists or not meta_exists:
    st.warning("⚠️ 未检测到知识库，请在左侧上传 PDF 并点击「构建知识库」")
    st.stop()

# ========== 渲染历史消息 ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 使用 markdown 渲染，同时允许 HTML
        st.markdown(message["content"], unsafe_allow_html=True)

        # 显示检索到的片段（如果有）
        if message["role"] == "assistant" and "contexts" in message:
            with st.expander("📖 查看检索到的相关片段"):
                for i, (content, score) in enumerate(message["contexts"]):
                    st.caption(f"片段 {i+1} (相关性: {score:.4f})")
                    st.text(content[:300] + "..." if len(content) > 300 else content)

# ========== 输入框 ==========
if prompt := st.chat_input("输入你的高数问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索并生成答案..."):
            try:
                if st.session_state.pipeline is None:
                    st.error("系统未初始化，请刷新页面")
                    st.stop()

                result = st.session_state.pipeline.ask(prompt)
                answer = result["answer"]

                # 显示答案
                st.markdown(answer, unsafe_allow_html=True)

                # 显示检索细节
                with st.expander("📖 查看检索到的相关片段"):
                    for i, (content, score) in enumerate(result["contexts"]):
                        st.caption(f"片段 {i+1} (相关性: {score:.4f})")
                        st.text(content[:300] + "..." if len(content) > 300 else content)

                # 保存消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "contexts": result["contexts"]
                })

                # ---------- 强制渲染 KaTeX（关键修复） ----------
                # 在消息更新后，调用 JavaScript 重新扫描并渲染所有公式
                st.components.v1.html(
                    """
                    <script>
                        // 等待 KaTeX 加载完成
                        function renderMath() {
                            if (typeof renderMathInElement !== 'undefined') {
                                try {
                                    renderMathInElement(document.body, {
                                        delimiters: [
                                            {left: '$$', right: '$$', display: true},
                                            {left: '\\(', right: '\\)', display: false},
                                            {left: '\\[', right: '\\]', display: true}
                                        ],
                                        throwOnError: false
                                    });
                                } catch (e) {
                                    console.warn('KaTeX render error:', e);
                                }
                            } else {
                                // 如果还没加载，延迟重试
                                setTimeout(renderMath, 500);
                            }
                        }
                        // 立即执行，并延迟执行确保 DOM 更新
                        setTimeout(renderMath, 100);
                        setTimeout(renderMath, 500);
                    </script>
                    """,
                    height=0,
                    scrolling=False
                )
                # ---------- 强制渲染结束 ----------

            except Exception as e:
                st.error(f"❌ 出错了: {e}")

# ========== 清空对话 ==========
if st.session_state.messages:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 在每次页面加载时，也触发一次渲染（针对已存在的消息）
# 但 st.components.v1.html 会在页面加载后执行，我们也可以放在这里
# 不过我们已经在消息更新时渲染了，所以就不重复执行了。