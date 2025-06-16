#!/bin/bash
# Health Check Script for World-Class Spam Filter
# Generated on 2025-06-16 12:47:51

echo "🏥 Health Check for World-Class Spam Filter"
echo "=" * 50

# Check Docker containers
if command -v docker &> /dev/null; then
    echo "🐳 Docker Container Status:"
    docker ps --filter "name=spam-filter" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
fi

# Check Kubernetes pods
if command -v kubectl &> /dev/null; then
    echo "☸️ Kubernetes Pod Status:"
    kubectl get pods -n spam-filter 2>/dev/null || echo "No Kubernetes deployment found"
fi

# Test endpoints
echo "🧪 Testing Endpoints:"

echo "Testing neural_network_advanced..."
curl -s -f http://localhost:8000/health && echo "✅ neural_network_advanced healthy" || echo "❌ neural_network_advanced unhealthy"

echo "Testing lightgbm_optimized..."
curl -s -f http://localhost:8001/health && echo "✅ lightgbm_optimized healthy" || echo "❌ lightgbm_optimized unhealthy"

echo "Testing xgboost_advanced..."
curl -s -f http://localhost:8002/health && echo "✅ xgboost_advanced healthy" || echo "❌ xgboost_advanced unhealthy"

echo "Testing ensemble_voting..."
curl -s -f http://localhost:8003/health && echo "✅ ensemble_voting healthy" || echo "❌ ensemble_voting unhealthy"

echo "Testing ensemble_stacking..."
curl -s -f http://localhost:8004/health && echo "✅ ensemble_stacking healthy" || echo "❌ ensemble_stacking unhealthy"
