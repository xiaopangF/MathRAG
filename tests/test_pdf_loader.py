"""
PDFLoader 单元测试
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import fitz
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


def create_layout_pdf(path: Path, page_count: int = 3) -> None:
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 30), "MathRAG Calculus Course")
        page.insert_text(
            (72, 110),
            f"Page {page_number} body content explains derivatives and integrals.",
        )
        page.insert_text((72, 820), "Internal teaching copy")
        page.insert_text((290, 820), str(page_number))
    document.save(path)
    document.close()


def create_image_only_pdf(path: Path, page_count: int = 1) -> None:
    source = fitz.open()
    source_page = source.new_page()
    source_page.insert_text(
        (72, 120),
        "Scanned calculus page about derivatives and integrals.",
        fontsize=18,
    )
    pixmap = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)

    document = fitz.open()
    for _ in range(page_count):
        page = document.new_page()
        page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    document.close()
    source.close()


def create_table_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    x_positions = [72, 220, 368]
    y_positions = [100, 140, 180]
    for x_position in x_positions:
        page.draw_line(
            (x_position, y_positions[0]),
            (x_position, y_positions[-1]),
        )
    for y_position in y_positions:
        page.draw_line(
            (x_positions[0], y_position),
            (x_positions[-1], y_position),
        )
    page.insert_text((82, 125), "Variable")
    page.insert_text((230, 125), "Value")
    page.insert_text((82, 165), "x")
    page.insert_text((230, 165), "x^2")
    page.wrap_contents()
    document.save(path)
    document.close()


def test_extract_pages_removes_repeated_headers_and_footers(tmp_path):
    pdf_path = tmp_path / "layout.pdf"
    create_layout_pdf(pdf_path)

    with PDFLoader(pdf_path) as loader:
        pages = loader.extract_pages()
        summary = loader.extraction_summary()

    assert len(pages) == 3
    assert all("MathRAG Calculus Course" not in page["text"] for page in pages)
    assert all("Internal teaching copy" not in page["text"] for page in pages)
    assert all("body content" in page["text"] for page in pages)
    assert summary["removed_margin_blocks"] >= 6
    assert summary["total_pages"] == 3
    assert summary["text_pages"] == 3


def test_extract_pages_records_positioned_blocks_and_two_column_order(tmp_path):
    pdf_path = tmp_path / "columns.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 120), "Left column starts here with a definition.")
    page.insert_text((72, 150), "Left column continues with its conditions.")
    page.insert_text((330, 120), "Right column starts with an example.")
    page.insert_text((330, 150), "Right column continues with the solution.")
    document.save(pdf_path)
    document.close()

    with PDFLoader(pdf_path) as loader:
        parsed_page = loader.extract_pages()[0]

    assert parsed_page["layout"] == "two_column"
    assert parsed_page["text"].index("Left column continues") < parsed_page[
        "text"
    ].index("Right column starts")
    assert all(len(block["bbox"]) == 4 for block in parsed_page["blocks"])
    assert [block["reading_order"] for block in parsed_page["blocks"]] == list(
        range(len(parsed_page["blocks"]))
    )


def test_textpage_is_extracted_with_its_original_page_instance(tmp_path):
    pdf_path = tmp_path / "textpage.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 120), "TextPage ownership regression test.")
    document.save(pdf_path)
    document.close()

    with PDFLoader(pdf_path) as loader:
        original_page = loader.doc[0]
        textpage = original_page.get_textpage()
        parsed = loader._extract_page_layout(
            0,
            page=original_page,
            textpage=textpage,
        )

    assert parsed["blocks"][0]["text"] == "TextPage ownership regression test."


def test_empty_pdf_page_has_diagnostics_and_requires_text_for_indexing(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    with PDFLoader(pdf_path) as loader:
        page = loader.extract_pages()[0]
        summary = loader.extraction_summary()
        with pytest.raises(ValueError, match="OCR"):
            loader.extract_full_text()

    assert page["quality"] == "empty"
    assert page["quality_flags"] == ["no_text"]
    assert page["needs_ocr"] is False
    assert summary["empty_pages"] == 1


def test_image_dominant_low_text_page_is_marked_for_ocr():
    quality, flags, needs_ocr = PDFLoader._page_quality(
        "short",
        image_count=1,
        image_coverage_ratio=0.9,
    )

    assert quality == "ocr_recommended"
    assert needs_ocr is True
    assert {"low_text", "image_dominant", "ocr_recommended"}.issubset(flags)


def test_image_only_page_is_an_ocr_candidate_when_ocr_is_disabled(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    create_image_only_pdf(pdf_path)

    with PDFLoader(pdf_path) as loader:
        page = loader.extract_pages()[0]
        summary = loader.extraction_summary()

    assert page["needs_ocr"] is True
    assert page["ocr_status"] == "recommended"
    assert summary["ocr_recommended_pages"] == [1]
    assert summary["ocr_applied_pages"] == []


def test_ocr_fallback_replaces_candidate_page_text(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    create_image_only_pdf(pdf_path)

    with PDFLoader(pdf_path, ocr_enabled=True) as loader:
        def fake_ocr(page_index, native_page):
            ocr_page = dict(native_page)
            ocr_page.update(
                {
                    "blocks": [
                        {
                            "text": "OCR derivative definition and calculation steps.",
                            "bbox": (72.0, 100.0, 500.0, 160.0),
                            "font_size": 12.0,
                            "font": "OCR",
                            "line_count": 1,
                            "position": "body",
                            "excluded_reason": None,
                            "reading_order": 0,
                            "role": "body",
                        }
                    ],
                    "layout": "single_column",
                    "extraction_method": "ocr",
                    "ocr_status": "applied",
                    "ocr_error": None,
                    "native_char_count": native_page["char_count"],
                }
            )
            return ocr_page

        monkeypatch.setattr(loader, "_extract_ocr_page", fake_ocr)
        page = loader.extract_pages()[0]
        summary = loader.extraction_summary()

    assert page["extraction_method"] == "ocr"
    assert page["ocr_status"] == "applied"
    assert page["needs_ocr"] is False
    assert "OCR derivative definition" in page["text"]
    assert "ocr_applied" in page["quality_flags"]
    assert summary["ocr_applied_pages"] == [1]
    assert summary["ocr_recommended_pages"] == []


def test_ocr_page_limit_and_failure_are_reported(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    create_image_only_pdf(pdf_path, page_count=2)

    with PDFLoader(pdf_path, ocr_enabled=True, ocr_max_pages=1) as loader:
        def fail_ocr(page_index, native_page):
            raise RuntimeError("tesseract unavailable")

        monkeypatch.setattr(loader, "_extract_ocr_page", fail_ocr)
        pages = loader.extract_pages()
        summary = loader.extraction_summary()

    assert pages[0]["ocr_status"] == "failed"
    assert pages[0]["ocr_error"] == "tesseract unavailable"
    assert pages[1]["ocr_status"] == "skipped_limit"
    assert summary["ocr_failed_pages"] == [1]
    assert summary["ocr_skipped_pages"] == [2]


def test_page_jsonl_and_extraction_summary_include_layout_metadata(tmp_path):
    pdf_path = tmp_path / "metadata.pdf"
    create_layout_pdf(pdf_path)
    pages_path = tmp_path / "pages.jsonl"
    summary_path = tmp_path / "summary.json"

    with PDFLoader(pdf_path) as loader:
        loader.save_pages_to_jsonl(pages_path)
        loader.save_extraction_summary(summary_path)

    first_page = json.loads(pages_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first_page["extraction_method"] == "native"
    assert first_page["layout"] in {"single_column", "two_column"}
    assert first_page["blocks"]
    assert summary["source"] == "metadata.pdf"
    assert summary["layouts"]


def test_table_detection_replaces_duplicate_cell_text_with_markdown(tmp_path):
    pdf_path = tmp_path / "table.pdf"
    create_table_pdf(pdf_path)

    with PDFLoader(pdf_path, table_detection_enabled=True) as loader:
        page = loader.extract_pages()[0]
        summary = loader.extraction_summary()

    assert page["table_count"] == 1
    assert page["tables"][0]["cells"] == [
        ["Variable", "Value"],
        ["x", "x^2"],
    ]
    assert "|Variable|Value|" in page["text"]
    assert page["text"].count("Variable") == 1
    assert any(block["role"] == "table" for block in page["blocks"])
    assert any(
        block["excluded_reason"] == "table_replaced"
        for block in page["blocks"]
    )
    assert summary["table_count"] == 1
    assert summary["table_pages"] == [1]


def test_formula_blocks_are_classified_and_counted(tmp_path):
    pdf_path = tmp_path / "formula.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 120), "f(x) = x^2 + 2*x + 1")
    document.save(pdf_path)
    document.close()

    with PDFLoader(pdf_path) as loader:
        page = loader.extract_pages()[0]
        summary = loader.extraction_summary()

    assert page["blocks"][0]["role"] == "formula_candidate"
    assert summary["formula_block_count"] == 1


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

    generated_pdf = base / ".pytest_tmp" / "integration" / "test.pdf"
    generated_pdf.parent.mkdir(parents=True, exist_ok=True)
    create_layout_pdf(generated_pdf, page_count=2)
    return generated_pdf


@pytest.mark.integration
def test_extract_pages_with_pdf():
    pdf_path = get_test_pdf_path()
    loader = PDFLoader(pdf_path)
    pages = loader.extract_pages()
    assert len(pages) > 0
    assert all("text" in p for p in pages)
    loader.close()


@pytest.mark.integration
def test_extract_full_text():
    pdf_path = get_test_pdf_path()
    loader = PDFLoader(pdf_path)
    full_text = loader.extract_full_text(add_page_marker=True)
    assert "[PAGE" in full_text
    assert len(full_text) > 0
    loader.close()


@pytest.mark.integration
def test_save_to_txt(tmp_path):
    pdf_path = get_test_pdf_path()
    loader = PDFLoader(pdf_path)
    out_file = tmp_path / "output.txt"
    loader.save_to_txt(out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").strip() != ""
    loader.close()
