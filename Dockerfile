# syntax=docker/dockerfile:1

# ---------- Builder stage ----------
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libta-lib0 libta-lib0-dev && rm -rf /var/lib/apt/lists/*

# Configure Poetry
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install "poetry==1.8.2"

WORKDIR /app

# Copy dependency manifests first for efficient caching
COPY pyproject.toml ./
# If lock file exists copy it; the command will ignore if absent
COPY poetry.lock* ./

RUN poetry install --no-dev --no-root

# Copy application source
COPY . .

# ---------- Runtime stage ----------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment and application from builder
COPY --from=builder /app /app

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]