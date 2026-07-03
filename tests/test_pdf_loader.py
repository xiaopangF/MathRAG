"""
PDFLoader 单元测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from src.loader.pdf_loader import PDFLoader  # 修正导入路径


# ========== 静态方法测试 ==========

def test_clean_text_basic():
    text = "Hello\r\nWorld\r\n\n\nExtra lines."
    expected = "Hello\nWorld\n\nExtra lines."
    assert PDFLoader.clean_text(text) == expected


def test_clean_text_preserve_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird."
    expected = "First paragraph.\n\nSecond paragraph.\n\nThird."
    assert PDFLoader.clean_text(text) == expected


def test_remove_page_noise_removes_plain_numbers():
    text = "123\nSome content\n- 45 -\nAnother line\n第 6 页"
    expected = "Some content\nAnother line"
    assert PDFLoader.remove_page_noise(text) == expected


def test_remove_page_noise_does_not_remove_non_page_lines():
    text = "Formula 1: x = 2\n3.14\nChapter 7\nPage 5 (reference)"
    expected = "Formula 1: x = 2\n3.14\nChapter 7\nPage 5 (reference)"
    assert PDFLoader.remove_page_noise(text) == expected


# ========== 集成测试（需要实际 PDF） ==========

def get_test_pdf_path() -> Path:
    base = project_root
    candidates = [
        base / "data" / "raw" / "sample.pdf",
        base / "data" / "raw" / "book.pdf",
        base / "tests" / "data" / "test.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@pytest.mark.integration
def test_extract_pages_with_pdf():
    pdf_path = get_test_pdf_path()
    if pdf_path is None:
        pytest.skip("没有可用的测试 PDF，跳过集成测试")
    loader = PDFLoader(pdf_path)
    pages = loader.extract_pages()
    assert len(pages) > 0
    assert all("text" in p for p in pages)
    loader.close()


@pytest.mark.integration
def test_extract_full_text():
    pdf_path = get_test_pdf_path()
    if pdf_path is None:
        pytest.skip("没有可用的测试 PDF，跳过集成测试")
    loader = PDFLoader(pdf_path)
    full_text = loader.extract_full_text(add_page_marker=True)
    assert "[PAGE" in full_text
    assert len(full_text) > 0
    loader.close()


@pytest.mark.integration
def test_save_to_txt(tmp_path):
    pdf_path = get_test_pdf_path()
    if pdf_path is None:
        pytest.skip("没有可用的测试 PDF，跳过集成测试")
    loader = PDFLoader(pdf_path)
    out_file = tmp_path / "output.txt"
    loader.save_to_txt(out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").strip() != ""
    loader.close()