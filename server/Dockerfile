# Multi-stage: dlib needs a full C++ toolchain + cmake to build, but only
# its compiled output needs to ship. This keeps the runtime image lean.

FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local

WORKDIR /app
COPY app/ app/
COPY models/ models/

ENV PORT=8080
EXPOSE 8080

# Cloud Run sends SIGTERM on scale-down; a single worker is fine for a
# proof of concept and keeps the model weights loaded once, not per-request.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
