from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_requirements_are_pinned():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    dependencies = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert dependencies
    assert all("==" in dependency for dependency in dependencies)


def test_compose_defines_backend_frontend_and_persistent_model_cache():
    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))

    assert compose["x-backend-environment"]["HF_ENDPOINT"] == "${HF_ENDPOINT:-https://huggingface.co}"
    assert compose["x-backend-environment"]["MATHRAG_LOG_JSON"] == "${MATHRAG_LOG_JSON:-true}"
    assert compose["x-backend-environment"]["MATHRAG_ALLOW_RUNTIME_API_KEY"] == "${MATHRAG_ALLOW_RUNTIME_API_KEY:-false}"
    assert compose["x-backend-environment"]["MATHRAG_JOB_MAX_ATTEMPTS"] == "${MATHRAG_JOB_MAX_ATTEMPTS:-3}"
    assert compose["x-backend-environment"]["MATHRAG_MAX_JSON_BODY_MB"] == "${MATHRAG_MAX_JSON_BODY_MB:-1}"
    assert compose["x-backend-environment"]["MATHRAG_PDF_OCR_ENABLED"] == "${MATHRAG_PDF_OCR_ENABLED:-true}"
    assert compose["x-backend-environment"]["MATHRAG_PDF_OCR_LANGUAGES"] == "${MATHRAG_PDF_OCR_LANGUAGES:-chi_sim+eng}"
    assert compose["x-backend-environment"]["MATHRAG_PDF_OCR_DPI"] == "${MATHRAG_PDF_OCR_DPI:-200}"
    assert compose["x-backend-environment"]["MATHRAG_PDF_OCR_MAX_PAGES"] == "${MATHRAG_PDF_OCR_MAX_PAGES:-100}"
    assert compose["x-backend-environment"]["MATHRAG_LLM_TIMEOUT_SECONDS"] == "${MATHRAG_LLM_TIMEOUT_SECONDS:-30}"
    assert compose["x-backend-environment"]["MATHRAG_LLM_MAX_RETRIES"] == "${MATHRAG_LLM_MAX_RETRIES:-2}"
    assert {"backend", "frontend", "model-cache"}.issubset(compose["services"])
    assert compose["services"]["frontend"]["depends_on"]["backend"]["condition"] == "service_healthy"
    assert compose["services"]["model-cache"]["profiles"] == ["tools"]
    assert compose["services"]["model-cache"]["command"] == ["python", "scripts/prewarm_models.py"]
    assert "huggingface-cache" in compose["volumes"]


def test_backend_dockerfile_installs_cpu_torch_before_runtime_requirements():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG TORCH_VERSION=2.6.0+cpu" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert dockerfile.index("torch==${TORCH_VERSION}") < dockerfile.index("-r requirements.txt")


def test_backend_dockerfile_installs_tesseract_ocr_languages():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "tesseract-ocr" in dockerfile
    assert "tesseract-ocr-eng" in dockerfile
    assert "tesseract-ocr-chi-sim" in dockerfile


def test_frontend_nginx_serves_built_static_assets():
    nginx_config = (PROJECT_ROOT / "frontend" / "nginx.conf").read_text(encoding="utf-8")

    assert "root /usr/share/nginx/html;" in nginx_config
    assert "index index.html;" in nginx_config
    assert "try_files $uri $uri/ /index.html;" in nginx_config


def test_model_prewarm_script_is_packaged_for_backend_image():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    prewarm_script = (PROJECT_ROOT / "scripts" / "prewarm_models.py").read_text(encoding="utf-8")

    assert "COPY scripts ./scripts" in dockerfile
    assert 'DEFAULT_HF_ENDPOINT = "https://huggingface.co"' in prewarm_script
    assert "SentenceTransformer(embedding_model)" in prewarm_script
    assert "CrossEncoder(reranker_model)" in prewarm_script


def test_ci_checks_backend_and_frontend():
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )

    assert set(workflow["jobs"]) == {"backend-tests", "frontend-build"}
    backend_steps = workflow["jobs"]["backend-tests"]["steps"]
    assert any(step.get("run") == "docker compose config --quiet" for step in backend_steps)
