"""
MathRAG PDF 文本加载器

功能：
1. 提取 PDF 全文
2. 保留页码信息
3. 清理基础噪声
4. 保存 TXT 和 JSONL
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF


class PDFLoader:
    """PDF 加载与文本提取器"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"不是 PDF 文件: {self.file_path}")
        self.doc = fitz.open(self.file_path)

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本，保留段落结构"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = []
        blank_count = 0
        for line in lines:
            if line == "":
                blank_count += 1
            else:
                if blank_count > 0:
                    cleaned.append("")
                    blank_count = 0
                cleaned.append(line)
        text = "\n".join(cleaned)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def remove_page_noise(text: str) -> str:
        """更保守的页码去除：只删除单独成行且只含数字或 '第X页' 的行"""
        lines = text.splitlines()
        kept = []
        for line in lines:
            stripped = line.strip()
            if re.fullmatch(r"\d{1,4}", stripped):
                continue
            if re.fullmatch(r"-\s*\d{1,4}\s*-", stripped):
                continue
            if re.fullmatch(r"第\s*\d{1,4}\s*页", stripped):
                continue
            kept.append(line)
        return "\n".join(kept)

    def extract_pages(self) -> list[dict]:
        pages = []
        for page_index in range(len(self.doc)):
            page = self.doc[page_index]
            text = page.get_text("text", sort=True)
            text = self.clean_text(text)
            text = self.remove_page_noise(text)
            if not text:
                continue
            pages.append({
                "source": self.file_path.name,
                "page": page_index + 1,
                "text": text,
                "char_count": len(text),
            })
        return pages

    def extract_full_text(self, add_page_marker: bool = True) -> str:
        pages = self.extract_pages()
        parts = []
        for page in pages:
            if add_page_marker:
                parts.append(f"\n\n[PAGE {page['page']}]\n{page['text']}")
            else:
                parts.append(page["text"])
        return "\n".join(parts).strip()

    def save_to_txt(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_text = self.extract_full_text(add_page_marker=True)
        output_path.write_text(full_text, encoding="utf-8")
        print(f"文本提取成功: {output_path}")
        print(f"页数: {len(self.doc)}, 字符数: {len(full_text)}")

    def save_pages_to_jsonl(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pages = self.extract_pages()
        with output_path.open("w", encoding="utf-8") as f:
            for page in pages:
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
        print(f"页面 JSONL 保存: {output_path}, 有效页数: {len(pages)}")