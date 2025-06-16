#!/bin/bash

# Large Files Diagnostic Script
# Date: 16/06/2025 19:00:00
# Purpose: Identify what's making the repository large

echo "=== REPOSITORY SIZE ANALYSIS ==="
echo ""

# Activate virtual environment
source spam_filter_env/bin/activate

echo "1. Current repository size:"
du -sh .git/
echo ""

echo "2. Working directory size (excluding .git and venv):"
du -sh --exclude='.git' --exclude='spam_filter_env' .
echo ""

echo "3. Largest files in working directory (>1MB):"
find . -type f -size +1M -not -path './.git/*' -not -path './spam_filter_env/*' -exec ls -lh {} \; | sort -k5hr | head -15
echo ""

echo "4. Files by extension size:"
echo "CSV files:"
find . -name "*.csv" -not -path './.git/*' -not -path './spam_filter_env/*' -exec ls -lh {} \; | sort -k5hr
echo ""
echo "Joblib files:"
find . -name "*.joblib" -not -path './.git/*' -not -path './spam_filter_env/*' -exec ls -lh {} \; | sort -k5hr
echo ""
echo "PKL files:"
find . -name "*.pkl" -not -path './.git/*' -not -path './spam_filter_env/*' -exec ls -lh {} \; | sort -k5hr
echo ""
echo "JSON files (>100KB):"
find . -name "*.json" -size +100k -not -path './.git/*' -not -path './spam_filter_env/*' -exec ls -lh {} \; | sort -k5hr
echo ""

echo "5. Directory sizes:"
du -sh */ 2>/dev/null | sort -hr | head -10
echo ""

echo "6. Git object count and size:"
git count-objects -vH
echo ""

echo "7. Recent commits with file changes:"
git log --oneline --stat -5
echo ""

echo "=== ANALYSIS COMPLETE ===" 