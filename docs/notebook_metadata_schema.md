# Notebook Metadata Schema

## Overview
This document defines the metadata schema for categorizing and organizing Jupyter notebooks in the project. Metadata helps users find relevant content and understand prerequisites.

## Metadata Fields

### Required Fields

#### 1. `tags` (array of strings)
Categories and topics covered in the notebook.

**Common Tags:**
- **Topic Tags**: 
  - `data-exploration`, `preprocessing`, `modeling`, `evaluation`, `deployment`
  - `vision-transformer`, `few-shot-learning`, `transfer-learning`
  - `network-security`, `malware-detection`, `packet-analysis`
  
- **Technical Tags**:
  - `pytorch`, `tensorflow`, `scikit-learn`
  - `visualization`, `statistics`, `feature-engineering`
  - `gpu-required`, `memory-intensive`
  
- **Process Tags**:
  - `eda`, `training`, `inference`, `optimization`
  - `debugging`, `profiling`, `testing`

#### 2. `difficulty` (string)
Indicates the technical level required.

**Values:**
- `beginner`: Basic Python and ML concepts
- `intermediate`: Familiarity with deep learning and PyTorch
- `advanced`: Expert knowledge of transformers and optimization

#### 3. `estimated_time` (string)
Approximate time to complete the notebook.

**Format**: `XX minutes` or `X hours`

**Guidelines:**
- Include time for reading explanations
- Account for code execution time
- Add buffer for exercises

#### 4. `prerequisites` (array of strings)
Knowledge or completed notebooks required.

**Examples:**
- `python-basics`, `pandas`, `numpy`
- `deep-learning-fundamentals`
- `01_environment_setup` (reference other notebooks)

### Optional Fields

#### 5. `gpu_required` (boolean)
Whether GPU acceleration is necessary.

#### 6. `dataset_size` (string)
Size of data processed in the notebook.

**Values**: `small` (<100MB), `medium` (100MB-1GB), `large` (>1GB)

#### 7. `outputs` (array of strings)
What the notebook produces.

**Examples:**
- `trained-model`, `visualizations`, `preprocessed-data`
- `performance-metrics`, `feature-importance`

## Metadata Examples

### Example 1: Beginner Data Exploration
```yaml
---
tags: [data-exploration, eda, network-security, visualization, pandas]
difficulty: beginner
estimated_time: 45 minutes
prerequisites: [python-basics, pandas, matplotlib]
dataset_size: small
outputs: [visualizations, statistics]
---
```

### Example 2: Advanced Model Training
```yaml
---
tags: [modeling, vision-transformer, pytorch, training, gpu-required]
difficulty: advanced
estimated_time: 3 hours
prerequisites: [deep-learning-fundamentals, 03_data_preprocessing, pytorch]
gpu_required: true
dataset_size: large
outputs: [trained-model, training-curves, checkpoints]
---
```

### Example 3: Few-Shot Learning Tutorial
```yaml
---
tags: [few-shot-learning, transfer-learning, tutorial, pytorch]
difficulty: intermediate
estimated_time: 90 minutes
prerequisites: [02_data_exploration, vision-transformer-basics]
gpu_required: false
outputs: [prototype-model, comparison-results]
---
```

## Tag Taxonomy

### Primary Categories
1. **Workflow Stage**
   - `setup` → `exploration` → `preprocessing` → `modeling` → `evaluation` → `deployment`

2. **Technical Domain**
   - `computer-vision`: Image processing, ViT concepts
   - `network-security`: Packet analysis, attack types
   - `machine-learning`: Algorithms, metrics, techniques

3. **Skill Level**
   - `tutorial`: Step-by-step learning
   - `reference`: Quick lookup and examples
   - `research`: Experimental and advanced topics

### Searchable Index

To find notebooks by topic, users can search for combinations:
- Beginner ViT: `tags: [vision-transformer, tutorial, beginner]`
- GPU Training: `tags: [training, gpu-required]`
- Data Prep: `tags: [preprocessing, data-engineering]`

## Implementation

### Adding Metadata to Notebooks

1. **New Notebooks**: Use template with metadata
2. **Existing Notebooks**: Add metadata as first cell
3. **Validation**: Run `validate_notebooks.py` to check

### Metadata Usage

1. **Documentation**: Auto-generate notebook index
2. **Search**: Enable tag-based discovery
3. **Learning Paths**: Create guided sequences
4. **Resource Planning**: Identify GPU/memory needs

## Best Practices

### Do's
- ✅ Use existing tags when possible
- ✅ Be specific with prerequisites
- ✅ Update time estimates based on feedback
- ✅ Include both broad and specific tags

### Don'ts
- ❌ Create redundant tags
- ❌ Underestimate time requirements
- ❌ Forget GPU requirements
- ❌ Use inconsistent tag formats

## Automated Tools

### Tag Extraction Script
```python
# Extract all metadata from notebooks
import json
from pathlib import Path

def extract_metadata(notebook_dir):
    metadata = {}
    for nb_path in Path(notebook_dir).glob('*.ipynb'):
        with open(nb_path) as f:
            nb = json.load(f)
            # Extract from first cell
            if nb['cells'] and nb['cells'][0]['cell_type'] == 'markdown':
                source = ''.join(nb['cells'][0]['source'])
                # Parse YAML frontmatter
                # ... parsing logic ...
    return metadata
```

### Notebook Index Generator
Generate an index of all notebooks with their metadata for easy navigation and search.