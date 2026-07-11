import json

from src.retriever.vector_indexer import load_chunks_from_folder


def test_chunk_loader_builds_math_search_text_and_preserves_display_text(tmp_path):
    chunk_dir = tmp_path / "children"
    chunk_dir.mkdir()
    chunk_file = chunk_dir / "child_0001_formula.txt"
    display_text = "∫₀¹ x² dx 表示定积分。"
    chunk_file.write_text(display_text, encoding="utf-8")
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "child_0001",
                "title": "定积分",
                "type": "formula",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    texts, metadata = load_chunks_from_folder(
        str(chunk_dir),
        str(metadata_path),
    )

    assert texts == [display_text]
    assert "定积分" in metadata[0]["search_text"]
    assert "integral" in metadata[0]["search_text"]
    assert "积分" in metadata[0]["search_text"]
    assert "^(2)" in metadata[0]["search_text"]
