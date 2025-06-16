# Dockerfile for neural_network_advanced v1.0.0
# Generated on 2025-06-16 12:47:51
# Model Type: neural_network
# F1 Capability: 94.7%

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
COPY models/neural_network_advanced_v1.0.0*.pth ./models/
COPY models/neural_network_advanced_vectorizer_v1.0.0.joblib ./models/
COPY production_serving_infrastructure.py .
COPY requirements.txt .

# Set environment variables
ENV MODEL_NAME=neural_network_advanced
ENV MODEL_TYPE=neural_network
ENV F1_CAPABILITY=94.67
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
