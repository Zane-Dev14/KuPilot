# ── Stage 1: build deps + pre-download embeddings ────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so container starts instantly
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', \
    model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached HuggingFace model
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY static/ static/
COPY templates/ templates/
COPY data/ data/

# Persistent volume mount point for Chroma DB + chat memory
VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIR=/tmp/chroma \
    MEMORY_PATH=/tmp/chat_memory.json \
    EMBEDDING_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
