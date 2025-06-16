#!/bin/bash

# BFG Repo-Cleaner Script (Safer Alternative)
# Date: 16/06/2025 19:00:00
# Purpose: Use BFG to clean git history safely

echo "=== BFG REPO-CLEANER SETUP ==="
echo "This is a safer alternative to git filter-branch"
echo ""

# Activate virtual environment
source spam_filter_env/bin/activate

echo "Step 1: Checking if BFG is available..."
if ! command -v bfg &> /dev/null; then
    echo "BFG not found. Installing..."
    if command -v java &> /dev/null; then
        echo "Java found. Downloading BFG..."
        wget -O bfg.jar https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar
        alias bfg='java -jar bfg.jar'
    else
        echo "Java not found. Please install Java first:"
        echo "sudo apt update && sudo apt install default-jre"
        echo "Then run this script again."
        exit 1
    fi
else
    echo "BFG found!"
fi

echo ""
echo "Step 2: Creating backup..."
git branch backup-before-bfg 2>/dev/null || echo "Backup branch already exists"

echo ""
echo "Step 3: Creating file patterns to remove..."
cat > large_files_to_remove.txt << EOF
*.joblib
*.pkl
*.zip
*.gz
*.tar
*.7z
EOF

echo ""
echo "Step 4: Running BFG cleanup..."
echo "Removing files larger than 10M..."
bfg --strip-blobs-bigger-than 10M .

echo "Removing specific file types..."
bfg --delete-files large_files_to_remove.txt .

echo ""
echo "Step 5: Cleaning up references..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "Step 6: Final size check..."
du -sh .git/

echo ""
echo "=== BFG CLEANUP COMPLETE ==="
echo "Repository history has been safely rewritten"
echo ""
echo "To push the cleaned repository:"
echo "git push --force-with-lease origin notebooks"

# Cleanup
rm -f large_files_to_remove.txt
rm -f bfg.jar 