# ── Stage 1: build deps + pre-download embeddings ────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --retries 10 --timeout 100 -r requirements.txt

RUN python - <<EOF
from langchain_huggingface import HuggingFaceEmbeddings
HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
EOF
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
COPY start.sh start.sh

# Make startup script executable
RUN chmod +x start.sh

# Persistent volume mount point for Chroma DB + chat memory
VOLUME ["/data"]

ENV PYTHONUNBUFFERED=1 \
    CHROMA_PERSIST_DIR=/tmp/chroma \
    MEMORY_PATH=/tmp/chat_memory.json \
    EMBEDDING_DEVICE=cpu

EXPOSE 8000

CMD ["./start.sh"]
