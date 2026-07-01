"""
MathRAG 向量索引构建器
功能：读取切分好的Chunks，用BGE模型转成向量，存入FAISS索引
"""
import os
import pickle
from pathlib import Path
from typing import List, Tuple

# 向量相关
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_chunks_from_folder(chunk_dir: str = "data/chunks") -> Tuple[List[str], List[str]]:
    """
    读取所有chunk文件，返回内容列表和标题列表
    """
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"❌ chunk目录不存在: {chunk_dir}")

    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.txt')]
    if not chunk_files:
        raise FileNotFoundError(f"❌ 在 {chunk_dir} 中没有找到任何 .txt 文件")

    contents = []
    titles = []

    for fname in sorted(chunk_files):
        fpath = os.path.join(chunk_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
            # 去掉开头的"标题: xxx"元信息行，只保留正文
            lines = text.split('\n')
            # 跳过第一行（标题）和分隔线
            content_lines = []
            skip_until_separator = True
            for line in lines:
                if skip_until_separator:
                    if line.startswith('='):
                        skip_until_separator = False
                    continue
                content_lines.append(line)
            content = '\n'.join(content_lines).strip()
            if content:
                contents.append(content)
                titles.append(fname.replace('.txt', ''))

    return contents, titles


def build_vector_index(
    chunk_dir: str = "data/chunks",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    index_path: str = "data/faiss_index",
    meta_path: str = "data/chunks_meta.pkl"
):
    """
    构建完整的向量索引
    """
    print("🚀 开始构建向量索引...")

    # 1. 加载所有chunk
    print(f"📂 正在读取: {chunk_dir}")
    contents, titles = load_chunks_from_folder(chunk_dir)
    print(f"📄 共加载 {len(contents)} 个知识块")

    # 2. 加载嵌入模型
    print(f"🤖 正在加载模型: {model_name}")
    model = SentenceTransformer(model_name)
    print("✅ 模型加载完成")

    # 3. 向量化所有内容
    print("⚡ 正在向量化（可能需要几分钟）...")
    embeddings = model.encode(contents, show_progress_bar=True)
    print(f"✅ 向量化完成，维度: {embeddings.shape}")

    # 4. 构建FAISS索引
    print("📦 正在构建FAISS索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积相似度
    # 归一化向量，使内积等价于余弦相似度
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"✅ FAISS索引构建完成，共 {index.ntotal} 个向量")

    # 5. 保存索引和元数据
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"✅ 索引已保存: {index_path}")

    # 保存元数据（标题和内容）
    with open(meta_path, "wb") as f:
        pickle.dump({"titles": titles, "contents": contents}, f)
    print(f"✅ 元数据已保存: {meta_path}")

    return index, embeddings, titles, contents


# ---------- 运行入口 ----------
if __name__ == "__main__":
    # 自动定位项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise RuntimeError("找不到项目根目录（包含 data 文件夹）")
        project_root = project_root.parent
    os.chdir(project_root)
    print(f"✅ 当前工作目录: {os.getcwd()}")

    # 构建索引
    build_vector_index()