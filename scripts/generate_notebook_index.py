#!/usr/bin/env python3
"""
Generate Notebook Index

This script scans all Jupyter notebooks and generates an index based on their metadata.
The index helps users discover notebooks by topic, difficulty, and prerequisites.

Usage:
    python generate_notebook_index.py
"""

import json
import yaml
from pathlib import Path
import re
from collections import defaultdict


def extract_metadata(notebook_path):
    """Extract metadata from notebook's first or second cell."""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
            
        if not notebook.get('cells'):
            return None
            
        # Check first two cells for metadata
        for i in range(min(2, len(notebook['cells']))):
            cell = notebook['cells'][i]
            if cell.get('cell_type') != 'markdown':
                continue
                
            source = ''.join(cell.get('source', []))
            
            # Look for YAML frontmatter
            yaml_match = re.match(r'---\n(.*?)\n---', source, re.DOTALL)
            if yaml_match:
                try:
                    metadata = yaml.safe_load(yaml_match.group(1))
                    metadata['filename'] = notebook_path.name
                    metadata['title'] = extract_title(notebook)
                    return metadata
                except yaml.YAMLError:
                    continue
                
    except (json.JSONDecodeError, FileNotFoundError):
        return None
        
    return None


def extract_title(notebook):
    """Extract notebook title from markdown headers."""
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            # Look for top-level header
            title_match = re.search(r'^# (.+)$', source, re.MULTILINE)
            if title_match:
                return title_match.group(1).strip()
    return "Untitled"


def generate_index(notebooks_dir):
    """Generate index from all notebooks in directory."""
    notebooks_dir = Path(notebooks_dir)
    metadata_list = []
    
    for notebook_path in sorted(notebooks_dir.glob('*.ipynb')):
        metadata = extract_metadata(notebook_path)
        if metadata:
            metadata_list.append(metadata)
        else:
            print(f"Warning: No metadata found in {notebook_path.name}")
            
    return metadata_list


def create_index_markdown(metadata_list):
    """Create markdown index from metadata."""
    output = ["# Notebook Index\n"]
    output.append("This index provides an overview of all notebooks in the project, organized by category and difficulty.\n")
    
    # Group by difficulty
    by_difficulty = defaultdict(list)
    for meta in metadata_list:
        difficulty = meta.get('difficulty', 'unspecified')
        by_difficulty[difficulty].append(meta)
        
    # Group by primary tag
    by_category = defaultdict(list)
    for meta in metadata_list:
        tags = meta.get('tags', [])
        if tags:
            primary_tag = tags[0]
            by_category[primary_tag].append(meta)
            
    # Output by difficulty
    output.append("\n## By Difficulty Level\n")
    for level in ['beginner', 'intermediate', 'advanced']:
        if level in by_difficulty:
            output.append(f"\n### {level.capitalize()}\n")
            for meta in by_difficulty[level]:
                output.append(format_notebook_entry(meta))
                
    # Output by category
    output.append("\n## By Category\n")
    categories = {
        'setup': 'Environment Setup',
        'data-exploration': 'Data Exploration',
        'preprocessing': 'Data Preprocessing',
        'modeling': 'Model Development',
        'evaluation': 'Model Evaluation',
        'deployment': 'Deployment',
        'tutorial': 'Tutorials'
    }
    
    for cat_key, cat_name in categories.items():
        if cat_key in by_category:
            output.append(f"\n### {cat_name}\n")
            for meta in by_category[cat_key]:
                output.append(format_notebook_entry(meta))
                
    # Complete notebook list
    output.append("\n## Complete Notebook List\n")
    output.append("\n| Notebook | Difficulty | Time | Tags | Prerequisites |")
    output.append("|----------|------------|------|------|---------------|")
    
    for meta in sorted(metadata_list, key=lambda x: x.get('filename', '')):
        filename = meta.get('filename', 'Unknown')
        title = meta.get('title', 'Untitled')
        difficulty = meta.get('difficulty', '-')
        time = meta.get('estimated_time', '-')
        tags = ', '.join(meta.get('tags', [])[:3])  # First 3 tags
        prereqs = ', '.join(meta.get('prerequisites', [])[:3])  # First 3 prereqs
        
        output.append(f"| [{title}](../notebooks/{filename}) | {difficulty} | {time} | {tags} | {prereqs} |")
        
    # Learning paths
    output.append("\n## Suggested Learning Paths\n")
    output.append("\n### Path 1: Beginner to Advanced\n")
    output.append("1. **Environment Setup** → 2. **Data Exploration** → 3. **Basic Preprocessing** → 4. **Simple Models** → 5. **Advanced Techniques**\n")
    
    output.append("\n### Path 2: Quick Start for ML Practitioners\n")
    output.append("1. **Data Overview** → 2. **Model Training** → 3. **Evaluation and Tuning**\n")
    
    # Resource requirements
    output.append("\n## Resource Requirements\n")
    gpu_notebooks = [m for m in metadata_list if m.get('gpu_required', False)]
    if gpu_notebooks:
        output.append("\n### GPU Required Notebooks:\n")
        for meta in gpu_notebooks:
            output.append(f"- {meta.get('title', 'Untitled')} ({meta.get('filename', '')})\n")
            
    return '\n'.join(output)


def format_notebook_entry(metadata):
    """Format a single notebook entry."""
    title = metadata.get('title', 'Untitled')
    filename = metadata.get('filename', 'unknown.ipynb')
    time = metadata.get('estimated_time', 'Not specified')
    tags = metadata.get('tags', [])
    
    entry = f"- **[{title}](../notebooks/{filename})** ({time})"
    if tags:
        entry += f" - Tags: {', '.join(tags[:3])}"
    entry += "\n"
    
    return entry


def create_tag_cloud(metadata_list):
    """Create tag frequency analysis."""
    tag_counts = defaultdict(int)
    
    for meta in metadata_list:
        for tag in meta.get('tags', []):
            tag_counts[tag] += 1
            
    output = ["## Tag Cloud\n"]
    output.append("Most frequently used tags:\n")
    
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        output.append(f"- `{tag}` ({count} notebooks)")
        
    return '\n'.join(output)


def main():
    """Main entry point."""
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    notebooks_dir = project_root / 'notebooks'
    docs_dir = project_root / 'docs'
    
    print(f"Scanning notebooks in: {notebooks_dir}")
    
    # Generate index
    metadata_list = generate_index(notebooks_dir)
    print(f"Found {len(metadata_list)} notebooks with metadata")
    
    # Create index markdown
    index_content = create_index_markdown(metadata_list)
    
    # Add tag cloud
    index_content += "\n\n" + create_tag_cloud(metadata_list)
    
    # Write index file
    index_path = docs_dir / 'notebook_index.md'
    with open(index_path, 'w') as f:
        f.write(index_content)
        
    print(f"Index generated: {index_path}")
    
    # Also create a JSON version for programmatic access
    json_path = docs_dir / 'notebook_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
        
    print(f"Metadata JSON: {json_path}")


if __name__ == '__main__':
    main()