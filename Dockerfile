# FROM python:3.11-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# # Install build deps (kept minimal)
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends gcc build-essential curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project
# COPY . /app

# # Ensure models are included in the image (Option B)
# COPY models /app/models

# # Use a non-root user
# RUN useradd -m appuser || true
# USER appuser

# EXPOSE 8000

# # Healthcheck
# HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://127.0.0.1:8000/health || exit 1

# # CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

# CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]





FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Ensure models are included
COPY models /app/models

# Use a non-root user
RUN useradd -m appuser || true
USER appuser

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://127.0.0.1:8000/health || exit 1

# Run FastAPI app
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
