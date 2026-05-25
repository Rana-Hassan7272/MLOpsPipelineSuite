# syntax=docker/dockerfile:1
# Multi-stage optimized build for MLOpsPipelineSuite
# Build stage - compile dependencies
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build deps only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage - minimal runtime
FROM python:3.11-slim-bookworm AS production

WORKDIR /app

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops \
    && mkdir -p /home/mlops /tmp/matplotlib \
    && chown -R mlops:mlops /home/mlops /tmp/matplotlib

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mlops/.local

# Set environment
ENV PATH=/home/mlops/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONPATH=/app

# Copy only necessary application files
COPY --chown=mlops:mlops algorithms/ ./algorithms/
COPY --chown=mlops:mlops api/ ./api/
COPY --chown=mlops:mlops configs/ ./configs/
COPY --chown=mlops:mlops data_pipeline/ ./data_pipeline/
COPY --chown=mlops:mlops evaluation/ ./evaluation/
COPY --chown=mlops:mlops experiments/ ./experiments/
COPY --chown=mlops:mlops monitoring/ ./monitoring/
COPY --chown=mlops:mlops retraining/ ./retraining/
COPY --chown=mlops:mlops tracking/ ./tracking/
COPY --chown=mlops:mlops training/ ./training/
COPY --chown=mlops:mlops results/ ./results/

# Create necessary directories
RUN mkdir -p datasets experiments/mlruns data \
    && chown -R mlops:mlops /app

USER mlops

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
