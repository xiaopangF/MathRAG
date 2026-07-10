from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.readiness_service import ReadinessService


def create_default_index(project_root: Path) -> None:
    index_dir = project_root / "data" / "faiss_index"
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"faiss")
    (index_dir / "chunks_meta.jsonl").write_text("{}\n", encoding="utf-8")


def test_readiness_is_ready_with_local_models_and_index(tmp_path, monkeypatch):
    create_default_index(tmp_path)
    embedding_dir = tmp_path / "models" / "embedding"
    reranker_dir = tmp_path / "models" / "reranker"
    embedding_dir.mkdir(parents=True)
    reranker_dir.mkdir(parents=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("MATHRAG_EMBEDDING_MODEL", str(embedding_dir))
    monkeypatch.setenv("MATHRAG_RERANKER_MODEL", str(reranker_dir))

    result = ReadinessService(tmp_path).inspect()

    assert result["status"] == "ready"
    assert result["can_answer_default"] is True
    assert result["can_build_index"] is True
    assert result["blockers"] == []


def test_readiness_explains_missing_runtime_dependencies(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MATHRAG_EMBEDDING_MODEL", "BAAI/test-embedding")
    monkeypatch.setenv("MATHRAG_RERANKER_MODEL", "BAAI/test-reranker")
    service = ReadinessService(tmp_path)
    monkeypatch.setattr(service, "_is_remote_model_cached", lambda model_name: False)

    result = service.inspect()

    assert result["status"] == "blocked"
    assert result["can_answer_default"] is False
    assert result["can_build_index"] is False
    assert result["checks"]["default_index"]["status"] == "missing"
    assert result["checks"]["embedding_model"]["status"] == "download_required"
    assert result["checks"]["reranker_model"]["status"] == "download_required"
    assert result["checks"]["deepseek_api_key"]["status"] == "missing"


def test_readiness_rejects_example_api_key_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")
    service = ReadinessService(tmp_path)
    monkeypatch.setattr(service, "_check_default_index", lambda: service._ready("index ready"))
    monkeypatch.setattr(service, "_check_model", lambda model_name, label: service._ready("model ready"))

    result = service.inspect()

    assert result["status"] == "blocked"
    assert result["checks"]["deepseek_api_key"]["status"] == "missing"


def test_readiness_detects_huggingface_snapshot_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "huggingface" / "hub"
    snapshot_dir = cache_root / "models--BAAI--test-model" / "snapshots" / "revision"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))

    result = ReadinessService(tmp_path)._check_model("BAAI/test-model", "测试模型")

    assert result["status"] == "ready"
    assert "缓存" in result["detail"]


def test_readiness_route_returns_component_statuses():
    response = TestClient(app).get("/api/readiness")

    assert response.status_code == 200
    assert set(response.json()["checks"]) == {
        "default_index",
        "embedding_model",
        "reranker_model",
        "deepseek_api_key",
    }


def test_local_dev_origin_is_allowed_on_alternate_port():
    response = TestClient(app).options(
        "/api/readiness",
        headers={
            "Origin": "http://127.0.0.1:5174",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"
