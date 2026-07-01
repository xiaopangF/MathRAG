"""
MathRAG 结构化切分器
功能：按章节标题将教材切成独立的知识块（Chunks）
"""
import os
import re
from pathlib import Path

def smart_split_by_titles(text: str) -> list:
    """
    按章节标题切分文本（支持多种标题格式）
    """
    # 定义常见标题模式
    patterns = [
        r'第[一二三四五六七八九十]+章\s*[^\n]+',      # "第一章 函数与极限"
        r'\d+\.\d+\s*[^\n]+',                        # "1.1 导数概念"
        r'§\s*\d+\.\d+\s*[^\n]+',                   # "§1.1 导数概念"
        r'（[一二三四五六七八九十]+）\s*[^\n]+',     # "（一）函数的概念"
    ]

    combined_pattern = '|'.join(patterns)
    # 使用原始字符串避免转义警告
    raw_chunks = re.split(r'(\n\s*)({})'.format(combined_pattern), text)

    chunks = []
    current_title = "开头"
    current_content = []

    for part in raw_chunks:
        if re.search(combined_pattern, part):
            if current_content:
                chunks.append({
                    "title": current_title,
                    "content": "".join(current_content).strip()
                })
                current_content = []
            current_title = part.strip()
        else:
            if part.strip():
                current_content.append(part)

    if current_content:
        chunks.append({
            "title": current_title,
            "content": "".join(current_content).strip()
        })

    # 如果没切出东西，按段落切
    if len(chunks) <= 1:
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            if p.strip():
                chunks.append({
                    "title": "正文段落",
                    "content": p.strip()
                })

    return chunks


def save_chunks_to_files(chunks: list, output_dir: str = "data/chunks"):
    """保存知识块到独立文件"""
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        # 清理标题中的非法字符
        raw_title = chunk['title']
        # 换行替换为空格
        clean_title = raw_title.replace('\n', ' ').replace('\r', '')
        # 过滤非法文件名字符
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            clean_title = clean_title.replace(ch, '_')
        # 截取前30个字符
        safe_title = clean_title[:30].strip()
        if not safe_title:
            safe_title = f"chunk_{i+1:04d}"

        filename = f"chunk_{i+1:04d}_{safe_title}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"标题: {chunk['title']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(chunk['content'])

    print(f"✅ 切分完成！共生成 {len(chunks)} 个知识块")
    print(f"📁 保存在: {output_dir}/")
    return chunks


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

    input_file = "data/processed/full_text.txt"
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    print(f"📖 原文总字符数: {len(full_text)}")

    chunks = smart_split_by_titles(full_text)
    save_chunks_to_files(chunks)

    print(f"\n📊 统计:")
    print(f"   - 总块数: {len(chunks)}")
    print(f"   - 平均字数: {sum(len(c['content']) for c in chunks) // len(chunks)}")
    print(f"   - 最大块: {max(len(c['content']) for c in chunks)}")
    print(f"   - 最小块: {min(len(c['content']) for c in chunks)}")