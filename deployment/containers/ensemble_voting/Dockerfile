# Dockerfile for ensemble_voting v1.0.0
# Generated on 2025-06-16 12:47:51
# Model Type: ensemble
# F1 Capability: 96.5%

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application code
COPY ensemble_models/*ensemble_voting*.joblib ./models/
COPY production_serving_infrastructure.py .
COPY requirements.txt .

# Set environment variables
ENV MODEL_NAME=ensemble_voting
ENV MODEL_TYPE=ensemble
ENV F1_CAPABILITY=96.5
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --retries=3 \
            CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Run the application
CMD ["python", "-m", "uvicorn", "production_serving_infrastructure:app", "--host", "0.0.0.0", "--port", "8000"]
