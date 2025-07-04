# Multi-stage Dockerfile optimized for Google Cloud Run
# This Dockerfile creates a production-ready image with minimal size and optimal performance

# Build stage - Install dependencies and build application
FROM python:3.11-slim as builder

# Set environment variables for build optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager for faster dependency resolution
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies using UV (faster than pip)
RUN uv sync --frozen --no-dev

# Production stage - Create minimal runtime image
FROM python:3.11-slim as runtime

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8080 \
    GOOGLE_GENAI_USE_VERTEXAI=true

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy Python environment from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port (Cloud Run uses PORT environment variable)
EXPOSE $PORT

# Set the default command
CMD ["python", "main.py"]