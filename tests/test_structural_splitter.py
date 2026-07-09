from src.splitter.structural_splitter import safe_filename_part, smart_split_by_titles


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
