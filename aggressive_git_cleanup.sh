#!/bin/bash

# Aggressive Git History Cleanup Script
# Date: 16/06/2025 19:00:00
# Purpose: Remove large files from entire git history

echo "=== AGGRESSIVE GIT CLEANUP ==="
echo "WARNING: This will rewrite git history!"
echo "Make sure you have a backup before proceeding."
echo ""

# Activate virtual environment
source spam_filter_env/bin/activate

echo "Step 1: Creating backup branch..."
git branch backup-before-cleanup 2>/dev/null || echo "Backup branch already exists"

echo ""
echo "Step 2: Removing large files from entire git history..."

# Remove specific large files from history
echo "Removing .joblib files from history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch models/**/*.joblib data/**/*.joblib' \
  --prune-empty --tag-name-filter cat -- --all

echo "Removing .pkl files from history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/features/*.pkl data/clean/features/*.pkl' \
  --prune-empty --tag-name-filter cat -- --all

echo "Removing zip files from history..."
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/*.zip data/**/*.zip *.zip' \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Step 3: Cleaning up git references..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Step 4: Force updating remote tracking..."
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin

echo ""
echo "Step 5: Final garbage collection..."
git gc --prune=now --aggressive

echo ""
echo "Step 6: Checking repository size..."
echo "Repository size:"
du -sh .git/
echo ""
echo "Largest remaining files in working directory:"
find . -type f -size +1M -not -path './.git/*' -not -path './spam_filter_env/*' | head -10

echo ""
echo "=== CLEANUP COMPLETE ==="
echo "Repository history has been rewritten"
echo "Next push will be much smaller"
echo ""
echo "To push the cleaned repository:"
echo "git push --force-with-lease origin notebooks"
echo ""
echo "WARNING: Use --force-with-lease carefully in shared repositories!" 