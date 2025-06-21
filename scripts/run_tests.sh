#!/bin/bash

# Script to run tests with coverage
set -e

echo "🧪 Running tests..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in Docker or local environment
if [ -f /.dockerenv ]; then
    echo "Running in Docker environment..."
    PYTHON_CMD=python
else
    echo "Running in local environment..."
    # Check if we should use docker-compose
    if command -v docker-compose &> /dev/null; then
        echo "Using docker-compose to run tests..."
        docker-compose exec -T dev-env bash -c "cd /workspace && bash scripts/run_tests.sh"
        exit $?
    else
        PYTHON_CMD=python3
    fi
fi

# Create test directories if they don't exist
mkdir -p tests/{unit,integration}
mkdir -p htmlcov

# Install test dependencies if needed
echo "Checking test dependencies..."
$PYTHON_CMD -m pip install -q pytest pytest-cov pytest-mock pytest-asyncio

# Run linting first
echo -e "\n${YELLOW}Running code quality checks...${NC}"

# Black formatting check
echo "Checking code formatting with Black..."
if $PYTHON_CMD -m black --check src/ tests/ 2>/dev/null; then
    echo -e "${GREEN}✓ Code formatting passed${NC}"
else
    echo -e "${RED}✗ Code formatting failed${NC}"
    echo "Run 'black src/ tests/' to fix formatting"
    exit 1
fi

# Flake8 linting
echo "Running Flake8..."
if $PYTHON_CMD -m flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503 2>/dev/null; then
    echo -e "${GREEN}✓ Flake8 linting passed${NC}"
else
    echo -e "${RED}✗ Flake8 linting failed${NC}"
    exit 1
fi

# Type checking with mypy
echo "Running type checking with mypy..."
if $PYTHON_CMD -m mypy src/ --ignore-missing-imports 2>/dev/null; then
    echo -e "${GREEN}✓ Type checking passed${NC}"
else
    echo -e "${YELLOW}⚠ Type checking warnings (non-blocking)${NC}"
fi

# Run unit tests
echo -e "\n${YELLOW}Running unit tests...${NC}"
if $PYTHON_CMD -m pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
    exit 1
fi

# Run integration tests if they exist
if [ -d "tests/integration" ] && [ "$(ls -A tests/integration/*.py 2>/dev/null)" ]; then
    echo -e "\n${YELLOW}Running integration tests...${NC}"
    if $PYTHON_CMD -m pytest tests/integration/ -v; then
        echo -e "${GREEN}✓ Integration tests passed${NC}"
    else
        echo -e "${RED}✗ Integration tests failed${NC}"
        exit 1
    fi
else
    echo -e "\n${YELLOW}No integration tests found, skipping...${NC}"
fi

# Generate coverage report
echo -e "\n${YELLOW}Coverage Report:${NC}"
$PYTHON_CMD -m coverage report

# Check coverage threshold
COVERAGE_THRESHOLD=80
COVERAGE=$(python -m coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')

if [ ! -z "$COVERAGE" ]; then
    if (( $(echo "$COVERAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then
        echo -e "${GREEN}✓ Coverage is ${COVERAGE}% (threshold: ${COVERAGE_THRESHOLD}%)${NC}"
    else
        echo -e "${RED}✗ Coverage is ${COVERAGE}% (below threshold: ${COVERAGE_THRESHOLD}%)${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}✅ All tests passed!${NC}"
echo "Coverage report available at: htmlcov/index.html"

# Run performance tests if they exist
if [ -d "tests/performance" ] && [ "$(ls -A tests/performance/*.py 2>/dev/null)" ]; then
    echo -e "\n${YELLOW}Running performance tests...${NC}"
    $PYTHON_CMD -m pytest tests/performance/ -v --benchmark-only
fi

# Check for security vulnerabilities
if command -v safety &> /dev/null; then
    echo -e "\n${YELLOW}Checking for security vulnerabilities...${NC}"
    safety check || echo -e "${YELLOW}⚠ Some security warnings found${NC}"
fi