"""
MathRAG PDF loader with page-aware layout metadata.

The native PyMuPDF path remains the fast default. It extracts positioned text
blocks, removes repeated margin noise, and records pages that may need OCR.
"""

import json
import re
from collections import Counter
from copy import deepcopy
from math import ceil
from pathlib import Path
from statistics import median
from typing import Any

import fitz  # PyMuPDF


class PDFLoader:
    """Extract searchable text and page-level diagnostics from a PDF."""

    MARGIN_RATIO = 0.12
    REPEATED_MARGIN_RATIO = 0.5
    MIN_REPEATED_MARGIN_PAGES = 3
    MIN_NATIVE_TEXT_CHARS = 40

    def __init__(
        self,
        file_path: str | Path,
        *,
        ocr_enabled: bool = False,
        ocr_languages: str = "chi_sim+eng",
        ocr_dpi: int = 200,
        ocr_max_pages: int = 100,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"不是 PDF 文件: {self.file_path}")
        if not re.fullmatch(r"[A-Za-z0-9_+-]{1,128}", ocr_languages):
            raise ValueError("OCR 语言代码格式无效")
        if not 72 <= ocr_dpi <= 600:
            raise ValueError("OCR DPI 必须在 72 到 600 之间")
        if not 1 <= ocr_max_pages <= 2000:
            raise ValueError("OCR 最大页数必须在 1 到 2000 之间")
        self.doc = fitz.open(self.file_path)
        self.ocr_enabled = ocr_enabled
        self.ocr_languages = ocr_languages
        self.ocr_dpi = ocr_dpi
        self.ocr_max_pages = ocr_max_pages
        self._pages_cache: list[dict[str, Any]] | None = None

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
        """Normalize whitespace while preserving paragraph boundaries."""
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
        """Remove standalone page numbers without deleting mathematical text."""
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

    @staticmethod
    def _noise_signature(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return re.sub(r"\d+", "<number>", normalized)

    @classmethod
    def _margin_position(
        cls,
        bbox: tuple[float, float, float, float],
        page_height: float,
    ) -> str:
        if bbox[3] <= page_height * cls.MARGIN_RATIO:
            return "header"
        if bbox[1] >= page_height * (1 - cls.MARGIN_RATIO):
            return "footer"
        return "body"

    @staticmethod
    def _order_band(
        blocks: list[dict[str, Any]],
        page_width: float,
    ) -> tuple[list[dict[str, Any]], bool]:
        if len(blocks) < 2:
            ordered = sorted(
                blocks,
                key=lambda item: (item["bbox"][1], item["bbox"][0]),
            )
            return ordered, False

        left = [
            item
            for item in blocks
            if (item["bbox"][0] + item["bbox"][2]) / 2 <= page_width * 0.48
        ]
        right = [
            item
            for item in blocks
            if (item["bbox"][0] + item["bbox"][2]) / 2 >= page_width * 0.52
        ]
        assigned = {id(item) for item in left + right}
        center = [item for item in blocks if id(item) not in assigned]
        if left and right and not center:
            key = lambda item: (item["bbox"][1], item["bbox"][0])
            return sorted(left, key=key) + sorted(right, key=key), True
        return sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])), False

    @classmethod
    def _order_blocks(
        cls,
        blocks: list[dict[str, Any]],
        page_width: float,
    ) -> tuple[list[dict[str, Any]], str]:
        if not blocks:
            return [], "empty"

        wide = sorted(
            [
                item
                for item in blocks
                if item["bbox"][2] - item["bbox"][0] >= page_width * 0.7
            ],
            key=lambda item: item["bbox"][1],
        )
        narrow = [item for item in blocks if item not in wide]
        ordered: list[dict[str, Any]] = []
        two_column = False
        previous_y = float("-inf")

        for wide_block in wide:
            boundary_y = (wide_block["bbox"][1] + wide_block["bbox"][3]) / 2
            band = [
                item
                for item in narrow
                if previous_y <= (item["bbox"][1] + item["bbox"][3]) / 2 < boundary_y
            ]
            band_ordered, band_two_column = cls._order_band(band, page_width)
            ordered.extend(band_ordered)
            ordered.append(wide_block)
            two_column = two_column or band_two_column
            previous_y = boundary_y

        remaining = [item for item in narrow if item not in ordered]
        band_ordered, band_two_column = cls._order_band(remaining, page_width)
        ordered.extend(band_ordered)
        two_column = two_column or band_two_column
        return ordered, "two_column" if two_column else "single_column"

    def _extract_page_layout(
        self,
        page_index: int,
        *,
        page: fitz.Page | None = None,
        textpage=None,
        extraction_method: str = "native",
    ) -> dict[str, Any]:
        page = page if page is not None else self.doc[page_index]
        text_kwargs = {"sort": True}
        if textpage is not None:
            text_kwargs["textpage"] = textpage
        page_dict = page.get_text("dict", **text_kwargs)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        text_blocks: list[dict[str, Any]] = []
        image_count = 0
        image_area = 0.0

        for raw_block in page_dict.get("blocks", []):
            bbox = tuple(
                round(float(value), 2)
                for value in raw_block.get("bbox", (0, 0, 0, 0))
            )
            if raw_block.get("type") != 0:
                image_count += 1
                image_area += max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                continue

            lines: list[str] = []
            font_sizes: list[float] = []
            fonts: list[str] = []
            for raw_line in raw_block.get("lines", []):
                spans = raw_line.get("spans", [])
                line_text = "".join(
                    str(span.get("text", "")) for span in spans
                ).strip()
                if line_text:
                    lines.append(line_text)
                font_sizes.extend(
                    float(span.get("size", 0.0))
                    for span in spans
                    if float(span.get("size", 0.0)) > 0
                )
                fonts.extend(
                    str(span.get("font", ""))
                    for span in spans
                    if span.get("font")
                )

            text = self.clean_text("\n".join(lines))
            if not text:
                continue
            text_blocks.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "font_size": round(max(font_sizes), 2) if font_sizes else None,
                    "font": Counter(fonts).most_common(1)[0][0] if fonts else "",
                    "line_count": len(lines),
                    "position": self._margin_position(bbox, page_height),
                    "excluded_reason": None,
                }
            )

        ordered_blocks, layout = self._order_blocks(text_blocks, page_width)
        body_font_sizes = [
            item["font_size"]
            for item in ordered_blocks
            if item["position"] == "body" and item["font_size"] is not None
        ]
        median_font_size = median(body_font_sizes) if body_font_sizes else 0.0
        for reading_order, block in enumerate(ordered_blocks):
            block["reading_order"] = reading_order
            is_heading = (
                block["position"] == "body"
                and block["font_size"] is not None
                and median_font_size > 0
                and block["font_size"] >= median_font_size * 1.2
                and len(block["text"]) <= 160
            )
            block["role"] = "heading_candidate" if is_heading else block["position"]

        page_area = max(page_width * page_height, 1.0)
        return {
            "source": self.file_path.name,
            "page": page_index + 1,
            "width": round(page_width, 2),
            "height": round(page_height, 2),
            "layout": layout,
            "blocks": ordered_blocks,
            "image_count": image_count,
            "image_coverage_ratio": round(min(image_area / page_area, 1.0), 4),
            "extraction_method": extraction_method,
            "ocr_status": "not_needed",
            "ocr_error": None,
        }

    @classmethod
    def _repeated_margin_signatures(
        cls,
        pages: list[dict[str, Any]],
    ) -> set[tuple[str, str]]:
        counts: Counter[tuple[str, str]] = Counter()
        for page in pages:
            seen: set[tuple[str, str]] = set()
            for block in page["blocks"]:
                if block["position"] not in {"header", "footer"}:
                    continue
                if block["line_count"] > 2 or len(block["text"]) > 160:
                    continue
                signature = cls._noise_signature(block["text"])
                if signature:
                    seen.add((block["position"], signature))
            counts.update(seen)

        threshold = max(
            cls.MIN_REPEATED_MARGIN_PAGES,
            ceil(len(pages) * cls.REPEATED_MARGIN_RATIO),
        )
        return {signature for signature, count in counts.items() if count >= threshold}

    @classmethod
    def _page_quality(
        cls,
        text: str,
        *,
        image_count: int,
        image_coverage_ratio: float,
    ) -> tuple[str, list[str], bool]:
        flags: list[str] = []
        char_count = len(text)
        if char_count == 0:
            flags.append("no_text")
        elif char_count < cls.MIN_NATIVE_TEXT_CHARS:
            flags.append("low_text")
        if "\ufffd" in text:
            flags.append("replacement_characters")

        image_dominant = image_count > 0 and image_coverage_ratio >= 0.5
        if image_dominant:
            flags.append("image_dominant")
        needs_ocr = image_count > 0 and (
            char_count == 0
            or (char_count < cls.MIN_NATIVE_TEXT_CHARS and image_dominant)
        )
        if needs_ocr:
            flags.append("ocr_recommended")

        if needs_ocr:
            quality = "ocr_recommended"
        elif char_count == 0:
            quality = "empty"
        elif flags:
            quality = "review"
        else:
            quality = "ok"
        return quality, flags, needs_ocr

    def _finalize_pages(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        repeated_signatures = self._repeated_margin_signatures(pages)
        for page in pages:
            kept_text: list[str] = []
            removed_count = 0
            for block in page["blocks"]:
                block["excluded_reason"] = None
                signature = (block["position"], self._noise_signature(block["text"]))
                if signature in repeated_signatures:
                    block["excluded_reason"] = f"repeated_{block['position']}"
                    removed_count += 1
                    continue
                kept_text.append(block["text"])

            text = self.clean_text(self.remove_page_noise("\n\n".join(kept_text)))
            ocr_applied = page["extraction_method"] == "ocr"
            quality, quality_flags, needs_ocr = self._page_quality(
                text,
                image_count=0 if ocr_applied else page["image_count"],
                image_coverage_ratio=(
                    0.0 if ocr_applied else page["image_coverage_ratio"]
                ),
            )
            if ocr_applied:
                quality_flags.append("ocr_applied")
            page.update(
                {
                    "text": text,
                    "char_count": len(text),
                    "text_block_count": sum(
                        block["excluded_reason"] is None for block in page["blocks"]
                    ),
                    "removed_margin_block_count": removed_count,
                    "quality": quality,
                    "quality_flags": quality_flags,
                    "needs_ocr": needs_ocr,
                }
            )
        return pages

    def _extract_ocr_page(
        self,
        page_index: int,
        native_page: dict[str, Any],
    ) -> dict[str, Any]:
        page = self.doc[page_index]
        textpage = page.get_textpage_ocr(
            language=self.ocr_languages,
            dpi=self.ocr_dpi,
            full=True,
        )
        ocr_page = self._extract_page_layout(
            page_index,
            page=page,
            textpage=textpage,
            extraction_method="ocr",
        )
        if not ocr_page["blocks"]:
            raise RuntimeError("OCR 未识别到可索引文本")
        ocr_page.update(
            {
                "image_count": native_page["image_count"],
                "image_coverage_ratio": native_page["image_coverage_ratio"],
                "native_char_count": native_page["char_count"],
                "ocr_status": "applied",
            }
        )
        return ocr_page

    def _apply_ocr_fallback(
        self,
        pages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = [index for index, page in enumerate(pages) if page["needs_ocr"]]
        if not candidates:
            return pages
        if not self.ocr_enabled:
            for index in candidates:
                pages[index]["ocr_status"] = "recommended"
            return pages

        for index in candidates[: self.ocr_max_pages]:
            native_page = pages[index]
            try:
                pages[index] = self._extract_ocr_page(index, native_page)
            except RuntimeError as exc:
                native_page["ocr_status"] = "failed"
                native_page["ocr_error"] = str(exc).strip()[:500]

        for index in candidates[self.ocr_max_pages :]:
            pages[index]["ocr_status"] = "skipped_limit"
        return pages

    def _build_pages(self) -> list[dict[str, Any]]:
        if self.doc is None:
            raise RuntimeError("PDF 文档已经关闭")

        pages = [self._extract_page_layout(index) for index in range(len(self.doc))]
        pages = self._finalize_pages(pages)
        pages = self._apply_ocr_fallback(pages)
        return self._finalize_pages(pages)

    def _get_pages(self) -> list[dict[str, Any]]:
        if self._pages_cache is None:
            self._pages_cache = self._build_pages()
        return self._pages_cache

    def extract_pages(self) -> list[dict[str, Any]]:
        return deepcopy(self._get_pages())

    def extraction_summary(self) -> dict[str, Any]:
        pages = self._get_pages()
        return {
            "source": self.file_path.name,
            "total_pages": len(pages),
            "text_pages": sum(bool(page["text"]) for page in pages),
            "empty_pages": sum(not page["text"] for page in pages),
            "ocr_recommended_pages": [
                page["page"] for page in pages if page["needs_ocr"]
            ],
            "ocr_applied_pages": [
                page["page"] for page in pages if page["ocr_status"] == "applied"
            ],
            "ocr_failed_pages": [
                page["page"] for page in pages if page["ocr_status"] == "failed"
            ],
            "ocr_skipped_pages": [
                page["page"]
                for page in pages
                if page["ocr_status"] == "skipped_limit"
            ],
            "ocr_enabled": self.ocr_enabled,
            "ocr_languages": self.ocr_languages,
            "ocr_dpi": self.ocr_dpi,
            "removed_margin_blocks": sum(
                page["removed_margin_block_count"] for page in pages
            ),
            "char_count": sum(page["char_count"] for page in pages),
            "layouts": dict(Counter(page["layout"] for page in pages)),
        }

    def extract_full_text(self, add_page_marker: bool = True) -> str:
        pages = self._get_pages()
        if not any(page["text"] for page in pages):
            raise ValueError("PDF 未提取到可索引文本，可能需要 OCR")

        parts = []
        for page in pages:
            if not page["text"]:
                continue
            if add_page_marker:
                parts.append(f"[PAGE {page['page']}]\n{page['text']}")
            else:
                parts.append(page["text"])
        return "\n\n".join(parts).strip()

    def save_to_txt(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_text = self.extract_full_text(add_page_marker=True)
        output_path.write_text(full_text, encoding="utf-8")
        summary = self.extraction_summary()
        print(f"文本提取成功: {output_path}")
        print(
            f"页数: {summary['total_pages']}, 文本页: {summary['text_pages']}, "
            f"字符数: {summary['char_count']}"
        )

    def save_pages_to_jsonl(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pages = self._get_pages()
        with output_path.open("w", encoding="utf-8") as file:
            for page in pages:
                file.write(json.dumps(page, ensure_ascii=False) + "\n")
        print(f"页面 JSONL 保存: {output_path}, 页面数: {len(pages)}")

    def save_extraction_summary(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.extraction_summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
