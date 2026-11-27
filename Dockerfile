# Production container for E-Commerce AI Customer Support Agent (FastAPI + SSE + SQLite + Hybrid RAG)
# Compatible with Docker Compose, Hugging Face Spaces (port 7860), Google Cloud Run, Render, and ECS.
FROM python:3.12-slim

ARG INSTALL_RAG=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    APP_MODE=auto \
    RETRIEVAL_MODE=hybrid \
    RATE_LIMIT_RPM=30 \
    DAILY_TOKEN_BUDGET=500000 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Create non-root user (UID 1000 matches Hugging Face Spaces & cloud container security standards)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first for better layer caching
COPY requirements.txt requirements-web.txt requirements-rag.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-web.txt \
    && if [ "$INSTALL_RAG" = "true" ]; then pip install -r requirements-rag.txt; fi

# Copy application source, static Web UI, skills, and synthetic seed datasets
COPY app.py web_server.py config.json ./
COPY customer_service ./customer_service
COPY web ./web
COPY data ./data
COPY skills ./skills
COPY runtime/.gitkeep ./runtime/.gitkeep

# Pre-build local Qdrant + FastEmbed Chinese embedding index at image build time when RAG is enabled
RUN mkdir -p /app/.cache/fastembed /app/runtime \
    && if [ "$INSTALL_RAG" = "true" ]; then python -m customer_service.knowledge_cli build; fi \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '7860'); urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=3)"

CMD ["python", "web_server.py"]
