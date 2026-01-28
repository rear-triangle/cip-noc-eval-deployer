FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY prompts/ ./prompts/

# Optional but recommended: make module imports reliable even if cwd changes
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.worker"]