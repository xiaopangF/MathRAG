from src.splitter.structural_splitter import safe_filename_part


def test_safe_filename_part_removes_path_separators_and_invalid_chars():
    title = '例如设" 为方程!,/0!12-.的全体实根所组成的集合，用'

    result = safe_filename_part(title)

    assert "/" not in result
    assert "\\" not in result
    assert '"' not in result
    assert result
