"""
MathRAG 向量索引构建器

功能：
1. 读取 child chunks
2. 使用 BGE 模型生成向量
3. 构建 FAISS 索引
4. 保存索引和 metadata
"""
import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.loader.math_text import build_math_search_text


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def configure_huggingface_endpoint() -> str:
    """Configure HuggingFace endpoint without overriding user settings."""
    endpoint = os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    return endpoint


def resolve_embedding_model(model_name: str | None = None) -> str:
    """Resolve embedding model from argument or environment."""
    return (
        model_name
        or os.getenv("MATHRAG_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )


def find_project_root() -> Path:
    """向上查找包含 data 文件夹的项目根目录"""
    current = Path(__file__).resolve().parent
    while not (current / "data").exists():
        if current.parent == current:
            raise RuntimeError("找不到项目根目录：需要包含 data 文件夹")
        current = current.parent
    return current


def load_metadata(metadata_path: Path) -> Dict[str, dict]:
    """读取 metadata.jsonl，按 id 建立索引"""
    if not metadata_path.exists():
        return {}
    metadata = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                metadata[item["id"]] = item
    return metadata


def extract_content_from_chunk_file(file_path: Path) -> str:
    """
    从 chunk txt 中提取正文。
    跳过头部信息，保留分隔线后的内容。
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    content_lines = []
    found_separator = False
    for line in lines:
        if not found_separator:
            # 分隔线可能是多个等号
            if line.strip().startswith("=="):
                found_separator = True
            continue
        content_lines.append(line)
    content = "\n".join(content_lines).strip()
    return content if content else text.strip()


def load_chunks_from_folder(
    chunk_dir: str = "data/chunks/children",
    metadata_path: str = "data/chunks/metadata.jsonl",
) -> Tuple[List[str], List[dict]]:
    """
    读取 child chunks，返回文本列表和 metadata 列表。
    """
    chunk_path = Path(chunk_dir)
    if not chunk_path.exists():
        raise FileNotFoundError(f"chunk 目录不存在: {chunk_dir}，请先运行切分脚本生成 chunks。")

    chunk_files = sorted(chunk_path.glob("*.txt"))
    if not chunk_files:
        raise FileNotFoundError(f"在 {chunk_path} 中没有找到任何 .txt 文件")

    metadata_map = load_metadata(Path(metadata_path))

    texts = []
    metadatas = []

    for file_path in chunk_files:
        # 从文件名提取 id，格式为 child_xxxx_...
        stem = file_path.stem
        parts = stem.split("_")
        if len(parts) < 2 or parts[0] != "child":
            warnings.warn(f"跳过非 child chunk 文件: {file_path.name}")
            continue
        chunk_id = f"{parts[0]}_{parts[1]}"  # 例如 child_0001

        content = extract_content_from_chunk_file(file_path)
        if not content:
            warnings.warn(f"文件内容为空，跳过: {file_path}")
            continue

        # 获取元数据，若缺失则构建默认值
        meta = metadata_map.get(chunk_id, {})
        if not meta:
            warnings.warn(f"未在 metadata.jsonl 中找到 id={chunk_id}，使用默认元数据")
            meta = {
                "id": chunk_id,
                "title": stem.replace("_", " "),
                "chapter": "",
                "section": "",
                "chunk_type": "text",
            }

        title = str(meta.get("title") or "").strip()
        retrieval_text = f"{title}\n{content}" if title else content
        meta.update({
            "file": str(file_path).replace("\\", "/"),
            "char_count": len(content),
            "search_text": build_math_search_text(retrieval_text),
        })

        texts.append(content)
        metadatas.append(meta)

    if not texts:
        raise ValueError("没有读取到有效 chunk 内容，请检查 chunk 目录和 metadata 文件。")

    return texts, metadatas


def build_vector_index(
    chunk_dir: str = "data/chunks/children",
    metadata_path: str = "data/chunks/metadata.jsonl",
    model_name: str | None = None,
    index_path: str = "data/faiss_index/index.faiss",
    output_meta_path: str = "data/faiss_index/chunks_meta.jsonl",
    batch_size: int = 32,
):
    """
    构建 FAISS 向量索引。
    """
    print("开始构建向量索引")
    print(f"读取 chunks: {chunk_dir}")
    texts, metadatas = load_chunks_from_folder(chunk_dir, metadata_path)
    print(f"共加载 {len(texts)} 个知识块")

    model_name = resolve_embedding_model(model_name)
    hf_endpoint = configure_huggingface_endpoint()
    print(f"加载模型: {model_name}")
    print(f"HuggingFace Endpoint: {hf_endpoint}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            "模型加载失败。请检查网络，或在 .env 中配置本地模型路径，例如：\n"
            "MATHRAG_EMBEDDING_MODEL=C:/models/bge-small-zh-v1.5\n"
            "也可以设置 HF_ENDPOINT=https://huggingface.co 或可用镜像。\n"
            f"原始错误: {e}"
        ) from e
    print("模型加载完成")

    print("正在向量化...")
    index_texts = [
        meta.get("search_text") or text
        for text, meta in zip(texts, metadatas)
    ]
    embeddings = model.encode(
        index_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    print(f"向量化完成，shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)

    ids = np.arange(len(texts), dtype="int64")
    index.add_with_ids(embeddings, ids)

    print(f"FAISS 索引构建完成，共 {index.ntotal} 个向量")

    # 保存索引和元数据
    index_path = Path(index_path)
    output_meta_path = Path(output_meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    with output_meta_path.open("w", encoding="utf-8") as f:
        for vector_id, (text, meta) in enumerate(zip(texts, metadatas)):
            record = {
                **meta,
                "vector_id": vector_id,
                "text": text,
                "search_text": meta.get("search_text") or text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 保存配置
    config_path = index_path.parent / "index_config.json"
    config = {
        "model_name": model_name,
        "dimension": dimension,
        "index_type": "IndexFlatIP + normalized embeddings",
        "chunk_count": len(texts),
        "math_search_text": True,
        "index_path": str(index_path).replace("\\", "/"),
        "metadata_path": str(output_meta_path).replace("\\", "/"),
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"索引已保存: {index_path}")
    print(f"元数据已保存: {output_meta_path}")
    print(f"配置已保存: {config_path}")

    return index, texts, metadatas


if __name__ == "__main__":
    try:
        project_root = find_project_root()
        os.chdir(project_root)
        print(f"当前工作目录: {os.getcwd()}")
        build_vector_index()
    except Exception as e:
        print(f"\n!!! 构建失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
