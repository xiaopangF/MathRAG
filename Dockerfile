# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_VERSION=2.6.0+cpu
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install --index-url "${TORCH_CPU_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && python -m pip install -r requirements.txt

COPY backend ./backend
COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY reports ./reports

RUN mkdir -p /app/data /app/storage

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
