#!/bin/bash
# Kubernetes Deployment Script for World-Class Spam Filter
# Generated on 2025-06-16 12:47:51

set -e

echo "🚀 Starting Kubernetes Deployment"
echo "=" * 50

# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Apply configmap
kubectl apply -f k8s/configmap.yaml

# Apply deployments

echo "Deploying neural_network_advanced..."
kubectl apply -f k8s/deployment-neural_network_advanced.yaml

echo "Deploying lightgbm_optimized..."
kubectl apply -f k8s/deployment-lightgbm_optimized.yaml

echo "Deploying xgboost_advanced..."
kubectl apply -f k8s/deployment-xgboost_advanced.yaml

echo "Deploying ensemble_voting..."
kubectl apply -f k8s/deployment-ensemble_voting.yaml

echo "Deploying ensemble_stacking..."
kubectl apply -f k8s/deployment-ensemble_stacking.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment --all -n spam-filter

# Check deployment status
echo "📊 Deployment Status:"
kubectl get pods -n spam-filter
kubectl get svc -n spam-filter
kubectl get ingress -n spam-filter

echo "✅ Kubernetes deployment completed!"
