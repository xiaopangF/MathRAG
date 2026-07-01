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

# ============== 读取 API Key（兼容本地和云端） ==============
# 注意：st.secrets 必须在导入 streamlit 之后才能使用

def get_api_key():
    """获取 DeepSeek API Key，优先从 st.secrets 读取，fallback 到 .env"""
    # 方法1：从 Streamlit Cloud Secrets 读取
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if key:
            print("✅ 从 Streamlit Secrets 读取 API Key 成功")
            return key
    except Exception:
        pass

    # 方法2：从 .env 文件读取（本地开发）
    try:
        from dotenv import load_dotenv
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            key = os.getenv("DEEPSEEK_API_KEY")
            if key:
                print(f"✅ 从 .env 文件读取 API Key 成功")
                return key
    except Exception:
        pass

    # 如果都没找到，返回 None
    return None

# 获取 API Key 并存入环境变量，方便其他模块读取
DEEPSEEK_API_KEY = get_api_key()
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
else:
    # 如果是在云端部署，缺失密钥会在这里提示
    if "st.secrets" in str(sys.modules.get("streamlit", {})):
        st.error("""
        ❌ 未找到 DEEPSEEK_API_KEY！
        
        请在 Streamlit Cloud 的 Secrets 中配置：
        1. 点击右下角「Manage app」
        2. 选择「Secrets」选项卡
        3. 添加：
           DEEPSEEK_API_KEY = "sk-你的真实密钥"
           DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
        """)
        st.stop()
    else:
        st.error("""
        ❌ 未找到 DEEPSEEK_API_KEY！
        
        本地运行：请在项目根目录创建 .env 文件，并写入：
        DEEPSEEK_API_KEY=sk-xxx
        DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
        """)
        st.stop()

# ... 后面是原来的导入和界面代码
# ============== 页面设置 ==============
st.set_page_config(
    page_title="MathRAG - 高数知识库问答",
    page_icon="📐",
    layout="wide"
)

# ============== 侧边栏：系统状态与操作 ==============
with st.sidebar:
    st.title("📐 MathRAG")
    st.caption("基于双阶段检索的高等数学知识库问答系统")

    st.divider()

    # 显示系统状态
    st.subheader("📊 系统状态")

    # 检查索引是否存在
    index_exists = Path("data/faiss_index").exists()
    meta_exists = Path("data/chunks_meta.pkl").exists()
    chunks_exist = Path("data/chunks").exists() and len(list(Path("data/chunks").glob("*.txt"))) > 0

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 知识块", len(list(Path("data/chunks").glob("*.txt"))) if chunks_exist else 0)
    col2.metric("🔍 索引", "✅" if index_exists else "❌")
    col3.metric("🧠 模型", "✅" if index_exists else "❌")

    st.divider()

    # ========== PDF上传与构建 ==========
    st.subheader("📤 上传教材")

    uploaded_file = st.file_uploader(
        "上传 PDF 文件（高等数学教材）",
        type=["pdf"],
        help="上传后点击下方按钮构建知识库"
    )

    if uploaded_file is not None:
        # 保存上传的PDF到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        st.success(f"✅ 已上传: {uploaded_file.name}")

        if st.button("🚀 构建知识库", type="primary", use_container_width=True):
            with st.spinner("正在处理PDF，请稍候..."):
                try:
                    # 1. 提取文本
                    loader = PDFLoader(pdf_path)
                    full_text = loader.extract_full_text()
                    loader.close()

                    # 2. 切分
                    chunks = smart_split_by_titles(full_text)
                    save_chunks_to_files(chunks, "data/chunks")

                    # 3. 构建向量索引（导入并运行）
                    from src.retriever.vector_indexer import build_vector_index
                    build_vector_index()

                    st.success("🎉 知识库构建完成！")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 构建失败: {e}")

        # 清理临时文件
        os.unlink(pdf_path)

    st.divider()

    st.caption("电子科技大学 · 人工智能专业 · 暑假项目")


# ============== 主区域：问答界面 ==============
st.title("📚 高等数学知识库问答")

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False

# 尝试初始化系统（如果索引存在）
if index_exists and meta_exists and not st.session_state.retriever_initialized:
    try:
        with st.spinner("🔄 正在加载系统..."):
            st.session_state.pipeline = MathRAGPipeline()
            st.session_state.retriever_initialized = True
            st.success("✅ 系统加载完成！可以开始提问了。")
    except Exception as e:
        st.error(f"❌ 系统加载失败: {e}")

# 如果没有索引，提示用户先上传
if not index_exists or not meta_exists:
    st.warning("⚠️ 未检测到知识库，请在左侧上传 PDF 并点击「构建知识库」")
    st.stop()

# ========== 显示历史消息 ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果是助手消息，展开显示检索细节
        if message["role"] == "assistant" and "contexts" in message:
            with st.expander("📖 查看检索到的相关片段"):
                for i, (content, score) in enumerate(message["contexts"]):
                    st.caption(f"片段 {i+1} (相关性: {score:.4f})")
                    st.text(content[:300] + "..." if len(content) > 300 else content)

# ========== 输入框 ==========
if prompt := st.chat_input("输入你的高数问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索并生成答案..."):
            try:
                if st.session_state.pipeline is None:
                    st.error("系统未初始化，请刷新页面")
                    st.stop()

                result = st.session_state.pipeline.ask(prompt)

                # 显示答案
                st.markdown(result["answer"])

                # 显示检索细节
                with st.expander("📖 查看检索到的相关片段"):
                    for i, (content, score) in enumerate(result["contexts"]):
                        st.caption(f"片段 {i+1} (相关性: {score:.4f})")
                        st.text(content[:300] + "..." if len(content) > 300 else content)

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": result["contexts"]
                })

            except Exception as e:
                st.error(f"❌ 出错了: {e}")


# ========== 底部：清空对话按钮 ==========
if st.session_state.messages:
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()