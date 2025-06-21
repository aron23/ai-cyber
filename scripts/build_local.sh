#!/bin/bash

# Script to build the project locally
set -e

echo "🔨 Building ML project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build timestamp
BUILD_TIME=$(date +"%Y-%m-%d %H:%M:%S")
BUILD_ID=$(date +"%Y%m%d%H%M%S")

echo -e "${BLUE}Build started at: ${BUILD_TIME}${NC}"
echo -e "${BLUE}Build ID: ${BUILD_ID}${NC}\n"

# Check if we're in Docker or local environment
if [ -f /.dockerenv ]; then
    echo "Building in Docker environment..."
    PYTHON_CMD=python
else
    echo "Building in local environment..."
    if command -v docker-compose &> /dev/null; then
        # Build Docker images first
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker-compose build
        
        # Run the build inside Docker
        echo -e "${YELLOW}Running build inside Docker...${NC}"
        docker-compose run --rm dev-env bash -c "cd /workspace && bash scripts/build_local.sh"
        exit $?
    else
        PYTHON_CMD=python3
    fi
fi

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
$PYTHON_CMD -m pip install -r requirements.txt

# Run tests first
echo -e "\n${YELLOW}Running tests...${NC}"
if bash scripts/run_tests.sh; then
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed - aborting build${NC}"
    exit 1
fi

# Create build directory
mkdir -p build/lib

# Package the source code
echo -e "\n${YELLOW}Packaging source code...${NC}"
cp -r src/ build/lib/
cp requirements.txt build/
cp README.md build/ 2>/dev/null || echo "# ML Project" > build/README.md

# Create setup.py if it doesn't exist
if [ ! -f setup.py ]; then
    echo -e "${YELLOW}Creating setup.py...${NC}"
    cat > setup.py << EOF
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ml-project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.9",
    author="ML Team",
    description="ML Project for Network Traffic Analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
    ],
)
EOF
fi

# Build Python package
echo -e "\n${YELLOW}Building Python package...${NC}"
$PYTHON_CMD setup.py sdist bdist_wheel

# Build Docker images for deployment
echo -e "\n${YELLOW}Building deployment Docker images...${NC}"

# Create production Dockerfile if it doesn't exist
if [ ! -f Dockerfile ]; then
    cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "src.api.main"]
EOF
fi

# Build production Docker image
if command -v docker &> /dev/null; then
    echo "Building production Docker image..."
    docker build -t ml-project:${BUILD_ID} -t ml-project:latest .
    echo -e "${GREEN}✓ Docker image built: ml-project:${BUILD_ID}${NC}"
fi

# Create build manifest
echo -e "\n${YELLOW}Creating build manifest...${NC}"
cat > build/manifest.json << EOF
{
    "build_id": "${BUILD_ID}",
    "build_time": "${BUILD_TIME}",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'not-in-git')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'not-in-git')",
    "python_version": "$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)",
    "platform": "$(uname -s)",
    "architecture": "$(uname -m)"
}
EOF

# Create build info
echo -e "\n${YELLOW}Build Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Build ID:        ${BUILD_ID}"
echo -e "Build Time:      ${BUILD_TIME}"
echo -e "Git Commit:      $(git rev-parse --short HEAD 2>/dev/null || echo 'not-in-git')"
echo -e "Python Version:  $($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# List build artifacts
echo -e "\n${YELLOW}Build artifacts:${NC}"
ls -la dist/ 2>/dev/null || echo "No dist/ directory found"
ls -la build/ | head -10

echo -e "\n${GREEN}✅ Build completed successfully!${NC}"
echo -e "Build artifacts are available in:"
echo -e "  - Python packages: dist/"
echo -e "  - Build files: build/"
if command -v docker &> /dev/null; then
    echo -e "  - Docker image: ml-project:${BUILD_ID}"
fi