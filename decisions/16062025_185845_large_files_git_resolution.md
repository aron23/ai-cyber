# Decision: Large Files Git Resolution Strategy
**Date:** 16/06/2025 18:58:45
**Decision ID:** large-files-git-resolution
**Status:** Proposed

## Problem Statement
Cannot push commits to git due to large files (model artifacts, feature files, zip files) that exceed git hosting size limits.

## Analysis
- **23+ .joblib model files** in various subdirectories not covered by current .gitignore patterns
- **6 .pkl feature files** in data directories
- **1 large zip file** (f45bkkt8pr-1.zip)
- Current .gitignore uses `models/*.joblib` but not `models/**/*.joblib` for subdirectories

## Decision
**Remove all large artifact files from git tracking and update .gitignore comprehensively**

### Rationale:
1. **Model files should not be in version control** - they are artifacts that can be regenerated
2. **Feature files can be rebuilt** from the feature engineering code
3. **Large datasets** should be stored externally or downloaded as needed
4. **Repository size optimization** improves clone/push performance

## Implementation
1. Updated `.gitignore` with comprehensive patterns:
   - `models/**/*.joblib` (catches all subdirectories)
   - `data/**/*.pkl` (all pickle files in data)
   - `*.zip` and other archive formats
2. Created `fix_large_files.sh` script to:
   - Remove files from git tracking (`git rm --cached`)
   - Commit changes
   - Run garbage collection
3. Preserved file structure - files remain on disk

## Impact Assessment
- ✅ **Positive:** Repository becomes pushable, faster clones
- ✅ **Positive:** Follows ML best practices (code not models in git)
- ⚠️ **Note:** Models need regeneration after fresh clone
- ✅ **Mitigation:** All models can be recreated from existing code

## Files to be Removed from Tracking
```
data/f45bkkt8pr-1.zip
models/neural_networks_series3/*.joblib
models/production_ensemble/*.joblib
models/baseline_models/*.joblib
models/notebook_artifacts/*.joblib
models/preprocessing_pipelines/*.joblib
data/features/*.pkl
data/clean/features/*.pkl
```

## Next Steps
1. Execute `fix_large_files.sh` script
2. Test git push capability
3. Document model regeneration process
4. Update team workflow to avoid committing large files

## Approval Status
Ready for implementation - script created and .gitignore updated. 