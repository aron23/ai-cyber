#!/bin/bash
# Docker Deployment Script for World-Class Spam Filter
# Generated on 2025-06-16 12:47:51

set -e

echo "🚀 Starting World-Class Spam Filter Deployment"
echo "=" * 50

# Build all containers
echo "🏗️ Building model containers..."

echo "Building neural_network_advanced..."
docker build -t spam-filter/neural_network_advanced:1.0.0 \
    -f Dockerfile.neural_network_advanced .

echo "Building lightgbm_optimized..."
docker build -t spam-filter/lightgbm_optimized:1.0.0 \
    -f Dockerfile.lightgbm_optimized .

echo "Building xgboost_advanced..."
docker build -t spam-filter/xgboost_advanced:1.0.0 \
    -f Dockerfile.xgboost_advanced .

echo "Building ensemble_voting..."
docker build -t spam-filter/ensemble_voting:1.0.0 \
    -f Dockerfile.ensemble_voting .

echo "Building ensemble_stacking..."
docker build -t spam-filter/ensemble_stacking:1.0.0 \
    -f Dockerfile.ensemble_stacking .

# Start services with docker-compose
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

curl -f http://localhost:8000/health || echo "⚠️ neural_network_advanced not healthy"

curl -f http://localhost:8001/health || echo "⚠️ lightgbm_optimized not healthy"

curl -f http://localhost:8002/health || echo "⚠️ xgboost_advanced not healthy"

curl -f http://localhost:8003/health || echo "⚠️ ensemble_voting not healthy"

curl -f http://localhost:8004/health || echo "⚠️ ensemble_stacking not healthy"

echo "✅ Deployment completed successfully!"
echo "📊 Access dashboard at: http://localhost/dashboard"
echo "📈 Access Grafana at: http://localhost:3000"
echo "🔍 Access Prometheus at: http://localhost:9090"
