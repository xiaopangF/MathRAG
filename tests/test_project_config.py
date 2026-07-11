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
    assert compose["x-backend-environment"]["MATHRAG_PDF_TABLE_DETECTION_ENABLED"] == "${MATHRAG_PDF_TABLE_DETECTION_ENABLED:-true}"
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


def test_ci_checks_supported_runtimes_and_docker_stack():
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )

    assert set(workflow["jobs"]) == {
        "backend-tests",
        "frontend-build",
        "docker-smoke",
    }

    backend_job = workflow["jobs"]["backend-tests"]
    matrix = backend_job["strategy"]["matrix"]["include"]
    runtimes = {(entry["os"], entry["python-version"]) for entry in matrix}
    assert runtimes == {
        ("ubuntu-latest", "3.11"),
        ("ubuntu-latest", "3.12"),
        ("windows-latest", "3.12"),
    }
    assert sum(entry["run-integration"] for entry in matrix) == 1

    backend_steps = backend_job["steps"]
    assert any(step.get("run") == "python -m pip check" for step in backend_steps)
    assert any(step.get("name") == "Run PDF integration tests" for step in backend_steps)
    upload_step = next(
        step for step in backend_steps if step.get("name") == "Upload backend test results"
    )
    assert upload_step["if"] == "always() && steps.unit-tests.outcome != 'skipped'"

    frontend_steps = workflow["jobs"]["frontend-build"]["steps"]
    assert any(step.get("name") == "Upload frontend build" for step in frontend_steps)

    smoke_job = workflow["jobs"]["docker-smoke"]
    smoke_steps = smoke_job["steps"]
    assert set(smoke_job["needs"]) == {"backend-tests", "frontend-build"}
    assert any(step.get("run") == "docker compose config --quiet" for step in smoke_steps)
    assert any("docker compose up" in step.get("run", "") for step in smoke_steps)
    assert any("127.0.0.1:5173/health" in step.get("run", "") for step in smoke_steps)
    assert any(
        step.get("run") == "docker compose down --remove-orphans"
        and step.get("if") == "always()"
        for step in smoke_steps
    )
