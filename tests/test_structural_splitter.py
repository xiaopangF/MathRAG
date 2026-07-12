import json

from src.splitter.structural_splitter import (
    safe_filename_part,
    save_chunks_to_files,
    smart_split_by_titles,
)


def test_safe_filename_part_removes_path_separators_and_invalid_chars():
    title = '例如设" 为方程!,/0!12-.的全体实根所组成的集合，用'

    result = safe_filename_part(title)

    assert "/" not in result
    assert "\\" not in result
    assert '"' not in result
    assert result


def test_smart_split_tracks_page_markers_without_keeping_marker_text():
    text = """[PAGE 12]
第二章 一元函数微分学
导数表示函数在某一点的变化率。
[PAGE 13]
定义 导数
导数的几何意义是切线斜率。
"""

    chunks = smart_split_by_titles(text)

    assert chunks[0]["title"] == "第二章 一元函数微分学"
    assert chunks[0]["page_start"] == 12
    assert chunks[0]["page_end"] == 12
    assert "[PAGE" not in chunks[0]["content"]
    assert chunks[1]["title"] == "定义 导数"
    assert chunks[1]["page_start"] == 13
    assert chunks[1]["page_end"] == 13


def test_smart_split_inherits_chapter_and_most_specific_section():
    text = """[PAGE 156]
第二章 一元函数微分学
本章研究导数。
二、微分中值定理
下面讨论中值定理。
（一）拉格朗日定理
设函数满足以下条件。
定理 1 拉格朗日中值定理
函数在闭区间连续，在开区间可导。
第三章 积分学
定义 1 原函数
若函数的导数等于被积函数，则称其为原函数。
"""

    chunks = smart_split_by_titles(text)

    assert [chunk["level"] for chunk in chunks] == [0, 1, 2, 3, 0, 3]
    assert chunks[0]["chapter"] == "第二章 一元函数微分学"
    assert chunks[0]["section"] == ""
    assert chunks[1]["chapter"] == "第二章 一元函数微分学"
    assert chunks[1]["section"] == "二、微分中值定理"
    assert chunks[2]["section"] == "（一）拉格朗日定理"
    assert chunks[3]["chapter"] == "第二章 一元函数微分学"
    assert chunks[3]["section"] == "（一）拉格朗日定理"
    assert chunks[4]["chapter"] == "第三章 积分学"
    assert chunks[4]["section"] == ""
    assert chunks[5]["chapter"] == "第三章 积分学"
    assert chunks[5]["section"] == ""


def test_smart_split_ignores_repeated_split_chapter_page_header():
    text = """[PAGE 155]
第二章
一元函数微分学
二、微分中值定理
定理 罗尔定理
函数在闭区间连续，在开区间可导。
[PAGE 156]
第二章
一元函数微分学
罗尔定理的几何意义是切线平行于横轴。
"""

    chunks = smart_split_by_titles(text)

    assert [chunk["title"] for chunk in chunks] == [
        "第二章 一元函数微分学",
        "二、微分中值定理",
        "定理 罗尔定理",
    ]
    assert chunks[-1]["page_start"] == 155
    assert chunks[-1]["page_end"] == 156
    assert chunks[-1]["chapter"] == "第二章 一元函数微分学"
    assert chunks[-1]["section"] == "二、微分中值定理"
    assert "罗尔定理的几何意义" in chunks[-1]["content"]


def test_smart_split_does_not_treat_bare_exercise_numbers_as_sections():
    text = """[PAGE 430]
第五章 多元函数微分学
七、多元函数的连续性
连续函数有最大最小值定理。
11.
这是习题编号，不是小节标题。
（4）
这也是编号。
"""

    chunks = smart_split_by_titles(text)

    assert [chunk["title"] for chunk in chunks] == [
        "第五章 多元函数微分学",
        "七、多元函数的连续性",
    ]
    assert "11." in chunks[-1]["content"]
    assert "（4）" in chunks[-1]["content"]
    assert chunks[-1]["section"] == "七、多元函数的连续性"


def test_saved_child_metadata_keeps_inherited_structure(tmp_path):
    chunks = smart_split_by_titles(
        """第二章 一元函数微分学
二、微分中值定理
（一）拉格朗日定理
定理 1 拉格朗日中值定理
函数在闭区间连续，在开区间可导。
"""
    )

    output_dir = tmp_path / "chunks"
    save_chunks_to_files(chunks, output_dir=str(output_dir), source_file="高等数学.pdf")
    records = [
        json.loads(line)
        for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    child = next(
        record
        for record in records
        if record["level"] == "child" and record["title"].startswith("定理")
    )

    assert child["chapter"] == "第二章 一元函数微分学"
    assert child["section"] == "（一）拉格朗日定理"
    assert child["source_file"] == "高等数学.pdf"
