"""
MathRAG 文档结构切分器
功能：
1. 智能识别章、节、小节标题
2. 按标题层级切分全文
3. 保存为父块和子块
4. 生成 metadata.jsonl
"""

import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 中文教材常见标题模式
CHAPTER_PATTERN = re.compile(
    r'^\s*第[一二三四五六七八九十百千万\d]+章\s*[^\n]{0,50}',
    re.MULTILINE
)
SECTION_PATTERN = re.compile(
    r'^\s*[一二三四五六七八九十百千万\d]+[、.．]\s*[^\n]{0,50}',
    re.MULTILINE
)
SECTION_SYMBOL_PATTERN = re.compile(
    r'^\s*§\s*\d+(?:\.\s*\d+)+\s*[^\n]{0,80}',
    re.MULTILINE,
)
SUBSECTION_PATTERN = re.compile(
    r'^\s*[（(][一二三四五六七八九十\d]+[）)]\s*[^\n]{0,50}',
    re.MULTILINE
)
UNIT_PATTERN = re.compile(
    r'^\s*(定义|定理|性质|公式|例题|例|证明|推论|命题|公理|引理|注|注意|习题|练习)\s*[^\n]{0,50}',
    re.MULTILINE
)
PAGE_MARKER_PATTERN = re.compile(r'^\s*\[PAGE\s+(\d+)\]\s*$', re.IGNORECASE)
CHAPTER_KEY_PATTERN = re.compile(r'第[一二三四五六七八九十百千万\d]+章')
CHAPTER_ONLY_PATTERN = re.compile(r'^\s*第[一二三四五六七八九十百千万\d]+章\s*$')
SECTION_PREFIX_PATTERN = re.compile(r'^\s*(?:[一二三四五六七八九十百千万\d]+[、.．]|[（(][一二三四五六七八九十\d]+[）)]|§\s*\d+(?:\.\s*\d+)+)\s*')
PAGE_DECORATION_PATTERN = re.compile(r'^[·\s\d+\-—–,，.。+!！#%&()*（）]+$')


def normalize_text(text: str) -> str:
    """标准化文本（换行、空白）"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def safe_filename_part(text: str, max_length: int = 40) -> str:
    """将标题转换为可用于 Windows/macOS/Linux 的安全文件名片段。"""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(". ")
    return (text[:max_length] or "untitled")


def apply_page_to_chunk(chunk: Dict[str, Any] | None, page: int | None) -> None:
    """Update a chunk's page range when a line is known to belong to a page."""
    if chunk is None or page is None:
        return
    if chunk.get("page_start") is None:
        chunk["page_start"] = page
    chunk["page_end"] = page


def chapter_key(title: str) -> str:
    """Return the stable chapter label, e.g. 第六章."""
    match = CHAPTER_KEY_PATTERN.search(title or "")
    return match.group(0) if match else ""


def has_meaningful_title_text(title: str) -> bool:
    """Reject bare exercise/list markers such as 11. or (4)."""
    stripped = SECTION_PREFIX_PATTERN.sub("", title or "").strip()
    if not stripped:
        return False
    if PAGE_DECORATION_PATTERN.fullmatch(stripped):
        return False
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z]', stripped))


def is_chapter_subtitle_line(line: str) -> bool:
    """Detect a short line that is likely the subtitle after a split chapter label."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 30:
        return False
    if PAGE_MARKER_PATTERN.match(stripped):
        return False
    if PAGE_DECORATION_PATTERN.fullmatch(stripped):
        return False
    if CHAPTER_PATTERN.match(stripped):
        return False
    if SECTION_PATTERN.match(stripped) or SECTION_SYMBOL_PATTERN.match(stripped):
        return False
    if SUBSECTION_PATTERN.match(stripped) or UNIT_PATTERN.match(stripped):
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', stripped))


def build_chapter_title(lines: list[str], index: int) -> tuple[str, int]:
    """
    Combine split chapter headers like:
    第二章
    一元函数微分学
    """
    title = lines[index].strip()
    consumed = 1
    if CHAPTER_ONLY_PATTERN.match(title) and index + 1 < len(lines):
        next_line = lines[index + 1].strip()
        if is_chapter_subtitle_line(next_line):
            title = f"{title} {next_line}"
            consumed = 2
    return title, consumed


def smart_split_by_titles(full_text: str) -> List[Dict[str, Any]]:
    """
    根据标题层级切分全文，返回 chunks 列表。
    每个 chunk 包含标题、内容、页码范围和当前章/节上下文。
    """
    text = normalize_text(full_text)
    lines = text.splitlines()

    chunks = []
    current_chunk = None
    current_page = None
    current_chapter = ""
    current_section = ""

    # 定义标题匹配函数
    def get_title_level(line: str) -> tuple:
        """返回 (匹配级别, 标题文本)，级别 0=章, 1=节, 2=小节, 3=单元, -1=无"""
        line = line.strip()
        if CHAPTER_PATTERN.match(line):
            return (0, line)
        if SECTION_PATTERN.match(line) and has_meaningful_title_text(line):
            return (1, line)
        if SECTION_SYMBOL_PATTERN.match(line) and has_meaningful_title_text(line):
            return (1, line)
        if SUBSECTION_PATTERN.match(line) and has_meaningful_title_text(line):
            return (2, line)
        if UNIT_PATTERN.match(line):
            return (3, line)
        return (-1, None)

    line_no = 0
    while line_no < len(lines):
        line = lines[line_no]
        page_match = PAGE_MARKER_PATTERN.match(line)
        if page_match:
            current_page = int(page_match.group(1))
            line_no += 1
            continue

        level, title = get_title_level(line)
        if level >= 0:
            consumed_lines = 1
            if level == 0:
                title, consumed_lines = build_chapter_title(lines, line_no)
                if current_chapter and chapter_key(title) == chapter_key(current_chapter):
                    line_no += consumed_lines
                    continue

            # 新标题开始，保存上一个chunk
            if current_chunk:
                chunks.append(current_chunk)
            if level == 0:
                current_chapter = title
                current_section = ""
            elif level in (1, 2):
                current_section = title
            current_chunk = {
                'title': title,
                'level': level,
                'content': '',
                'start_line': line_no,
                'page_start': current_page,
                'page_end': current_page,
                'chapter': current_chapter,
                'section': current_section,
            }
            line_no += consumed_lines
        else:
            # 普通内容行
            if current_chunk:
                current_chunk['content'] += line + '\n'
                apply_page_to_chunk(current_chunk, current_page)
            else:
                # 还没有标题，全部归入"前言"
                if not chunks or chunks[-1]['title'] != '前言':
                    current_chunk = {
                        'title': '前言',
                        'level': -1,
                        'content': '',
                        'start_line': 0,
                        'page_start': current_page,
                        'page_end': current_page,
                        'chapter': '',
                        'section': '',
                    }
                current_chunk['content'] += line + '\n'
                apply_page_to_chunk(current_chunk, current_page)
            line_no += 1

    # 保存最后一个
    if current_chunk:
        chunks.append(current_chunk)

    # 清理内容首尾空行
    for chunk in chunks:
        chunk['content'] = chunk['content'].strip()

    return chunks


def save_chunks_to_files(
    chunks: List[Dict[str, Any]],
    output_dir: str = "data/chunks",
    clear_existing: bool = True,
    source_file: str = "",
):
    """
    将切分结果保存为文件，并生成 metadata.jsonl
    """
    output_path = Path(output_dir)
    if clear_existing and output_path.exists():
        shutil.rmtree(output_path)

    parent_dir = output_path / "parents"
    child_dir = output_path / "children"
    parent_dir.mkdir(parents=True, exist_ok=True)
    child_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path / "metadata.jsonl"

    # 保存父块（按章/节）
    parent_id = 0
    child_id = 0

    with metadata_path.open('w', encoding='utf-8') as meta_file:
        for chunk in chunks:
            # 生成父块
            parent_id += 1
            parent = {
                'id': f'parent_{parent_id:04d}',
                'title': chunk['title'],
                'level': chunk['level'],
                'content': chunk['content'],
                'start_line': chunk.get('start_line', 0),
                'page_start': chunk.get('page_start'),
                'page_end': chunk.get('page_end'),
                'chapter': chunk.get('chapter', ''),
                'section': chunk.get('section', ''),
            }
            parent_file = parent_dir / f"{parent['id']}_{safe_filename_part(parent['title'])}.txt"
            parent_file.write_text(parent['content'], encoding='utf-8')

            # 记录父块元数据
            meta_file.write(json.dumps({
                'id': parent['id'],
                'level': 'parent',
                'title': parent['title'],
                'file': str(parent_file.relative_to(output_path.parent)),
                'char_count': len(parent['content']),
                'source_file': source_file,
                'page_start': parent['page_start'],
                'page_end': parent['page_end'],
                'chapter': parent['chapter'],
                'section': parent['section'],
            }, ensure_ascii=False) + '\n')

            # 对父块内容进行子块切分（按单元）
            unit_matches = list(UNIT_PATTERN.finditer(parent['content']))
            if unit_matches:
                # 提取单元
                for i, match in enumerate(unit_matches):
                    unit_title = match.group(0).strip()
                    start = match.end()
                    end = unit_matches[i + 1].start() if i + 1 < len(unit_matches) else len(parent['content'])
                    unit_content = parent['content'][start:end].strip()
                    if unit_content:
                        child_id += 1
                        child = {
                            'id': f'child_{child_id:04d}',
                            'parent_id': parent['id'],
                            'title': unit_title,
                            'content': unit_content,
                            'type': classify_unit(unit_title),
                            'page_start': parent['page_start'],
                            'page_end': parent['page_end'],
                            'chapter': parent['chapter'],
                            'section': parent['section'],
                        }
                        child_file = child_dir / f"{child['id']}_{safe_filename_part(child['title'])}.txt"
                        child_file.write_text(child['content'], encoding='utf-8')

                        meta_file.write(json.dumps({
                            'id': child['id'],
                            'level': 'child',
                            'parent_id': child['parent_id'],
                            'title': child['title'],
                            'type': child['type'],
                            'file': str(child_file.relative_to(output_path.parent)),
                            'char_count': len(child['content']),
                            'source_file': source_file,
                            'page_start': child['page_start'],
                            'page_end': child['page_end'],
                            'chapter': child['chapter'],
                            'section': child['section'],
                        }, ensure_ascii=False) + '\n')
            else:
                # 没有单元，整个父块作为一个子块
                child_id += 1
                child = {
                    'id': f'child_{child_id:04d}',
                    'parent_id': parent['id'],
                    'title': parent['title'],
                    'content': parent['content'],
                    'type': 'text',
                    'page_start': parent['page_start'],
                    'page_end': parent['page_end'],
                    'chapter': parent['chapter'],
                    'section': parent['section'],
                }
                child_file = child_dir / f"{child['id']}_{safe_filename_part(child['title'])}.txt"
                child_file.write_text(child['content'], encoding='utf-8')

                meta_file.write(json.dumps({
                    'id': child['id'],
                    'level': 'child',
                    'parent_id': child['parent_id'],
                    'title': child['title'],
                    'type': child['type'],
                    'file': str(child_file.relative_to(output_path.parent)),
                    'char_count': len(child['content']),
                    'source_file': source_file,
                    'page_start': child['page_start'],
                    'page_end': child['page_end'],
                    'chapter': child['chapter'],
                    'section': child['section'],
                }, ensure_ascii=False) + '\n')

    print(f"切分完成：父块 {parent_id} 个，子块 {child_id} 个")
    print(f"输出目录：{output_path}")


def classify_unit(title: str) -> str:
    """根据标题判断单元类型"""
    mapping = {
        "定义": "definition",
        "定理": "theorem",
        "性质": "property",
        "公式": "formula",
        "例题": "example",
        "例": "example",
        "证明": "proof",
        "推论": "corollary",
        "命题": "proposition",
        "公理": "axiom",
        "引理": "lemma",
        "注": "note",
        "注意": "note",
        "习题": "exercise",
        "练习": "exercise",
    }
    for key, value in mapping.items():
        if title.startswith(key):
            return value
    return "text"


# 快速测试（可选）
if __name__ == "__main__":
    # 读取 full_text.txt 测试
    test_file = Path("data/processed/full_text.txt")
    if test_file.exists():
        text = test_file.read_text(encoding="utf-8")
        chunks = smart_split_by_titles(text)
        print(f"共切分出 {len(chunks)} 个块")
        for i, chunk in enumerate(chunks[:5]):
            print(f"{i + 1}. {chunk['title']} ({chunk['level']}) - {len(chunk['content'])} 字符")
        # 保存
        save_chunks_to_files(chunks)
    else:
        print("请先运行 PDF 提取生成 data/processed/full_text.txt")
