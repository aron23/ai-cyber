#!/usr/bin/env python3
"""
Notebook Validation Script

This script validates Jupyter notebooks for quality and reproducibility standards.
It checks for proper structure, metadata, and ensures notebooks can run without errors.

Usage:
    python validate_notebooks.py [notebook_path]
    python validate_notebooks.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import re
import tempfile
import shutil


class NotebookValidator:
    """Validates Jupyter notebooks against project standards."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_notebook(self, notebook_path):
        """Validate a single notebook file."""
        print(f"\nValidating: {notebook_path}")
        self.errors = []
        self.warnings = []
        
        # Check file exists
        if not notebook_path.exists():
            self.errors.append(f"File not found: {notebook_path}")
            return False
            
        # Load notebook
        try:
            with open(notebook_path, 'r') as f:
                notebook = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False
            
        # Run validation checks
        self._check_metadata(notebook)
        self._check_structure(notebook)
        self._check_code_quality(notebook)
        self._check_reproducibility(notebook)
        self._check_execution(notebook_path)
        
        # Report results
        self._report_results(notebook_path)
        
        return len(self.errors) == 0
        
    def _check_metadata(self, notebook):
        """Check for required metadata in first cell."""
        if not notebook.get('cells'):
            self.errors.append("Notebook has no cells")
            return
            
        first_cell = notebook['cells'][0]
        if first_cell.get('cell_type') != 'markdown':
            self.warnings.append("First cell should be markdown with metadata")
            return
            
        source = ''.join(first_cell.get('source', []))
        
        # Check for YAML frontmatter
        if not source.strip().startswith('---'):
            self.warnings.append("Missing YAML frontmatter with metadata")
            return
            
        # Extract and validate metadata
        yaml_match = re.match(r'---\n(.*?)\n---', source, re.DOTALL)
        if yaml_match:
            metadata = yaml_match.group(1)
            required_fields = ['tags', 'difficulty', 'estimated_time', 'prerequisites']
            for field in required_fields:
                if field not in metadata:
                    self.warnings.append(f"Missing metadata field: {field}")
                    
    def _check_structure(self, notebook):
        """Check notebook structure and organization."""
        cells = notebook.get('cells', [])
        
        # Check for title
        has_title = False
        has_overview = False
        has_objectives = False
        
        for cell in cells[:10]:  # Check first 10 cells
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', [])).lower()
                if '# ' in source and not has_title:
                    has_title = True
                if 'overview' in source:
                    has_overview = True
                if 'objective' in source or 'learning' in source:
                    has_objectives = True
                    
        if not has_title:
            self.warnings.append("Missing clear title")
        if not has_overview:
            self.warnings.append("Missing overview section")
        if not has_objectives:
            self.warnings.append("Missing learning objectives")
            
    def _check_code_quality(self, notebook):
        """Check code cells for quality issues."""
        cells = notebook.get('cells', [])
        
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Check for imports
                if i == 0 and 'import' not in source:
                    self.warnings.append("First code cell should contain imports")
                    
                # Check for hardcoded paths
                if '/home/' in source or 'C:\\' in source or 'Users/' in source:
                    self.errors.append(f"Cell {i}: Contains hardcoded absolute path")
                    
                # Check for print statements explaining output
                if len(source.strip()) > 0 and 'print' not in source and 'display' not in source:
                    # Check if there's explanation in surrounding markdown
                    if i > 0 and cells[i-1].get('cell_type') != 'markdown':
                        self.warnings.append(f"Cell {i}: Code cell without explanation")
                        
    def _check_reproducibility(self, notebook):
        """Check for reproducibility issues."""
        cells = notebook.get('cells', [])
        
        has_seed = False
        has_imports = False
        
        for cell in cells:
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Check for random seed
                if 'random.seed' in source or 'np.random.seed' in source or 'torch.manual_seed' in source:
                    has_seed = True
                    
                # Check for imports
                if 'import' in source:
                    has_imports = True
                    
        if not has_imports:
            self.errors.append("No import statements found")
            
        # Check if notebook uses random operations
        full_source = ''.join([''.join(cell.get('source', [])) 
                              for cell in cells if cell.get('cell_type') == 'code'])
        if ('random' in full_source or 'shuffle' in full_source or 'sample' in full_source) and not has_seed:
            self.warnings.append("Uses random operations without setting seed")
            
    def _check_execution(self, notebook_path):
        """Check if notebook executes without errors."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_notebook = Path(temp_dir) / notebook_path.name
            shutil.copy(notebook_path, temp_notebook)
            
            # Try to execute notebook
            try:
                result = subprocess.run(
                    ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                     '--ExecutePreprocessor.timeout=300', str(temp_notebook)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    self.errors.append(f"Notebook execution failed: {result.stderr}")
                    
            except subprocess.CalledProcessError as e:
                self.errors.append(f"Failed to execute notebook: {e}")
            except FileNotFoundError:
                self.warnings.append("Jupyter not found - skipping execution test")
                
    def _report_results(self, notebook_path):
        """Report validation results."""
        print(f"\nResults for {notebook_path.name}:")
        print("-" * 50)
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   - {error}")
                
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
                
        if not self.errors and not self.warnings:
            print("✅ All checks passed!")
            
        print("-" * 50)
        

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Validate Jupyter notebooks')
    parser.add_argument('notebook', nargs='?', help='Path to notebook file')
    parser.add_argument('--all', action='store_true', help='Validate all notebooks')
    
    args = parser.parse_args()
    
    validator = NotebookValidator()
    
    if args.all:
        # Find all notebooks
        notebooks_dir = Path(__file__).parent.parent / 'notebooks'
        notebooks = list(notebooks_dir.glob('*.ipynb'))
        
        if not notebooks:
            print("No notebooks found in notebooks/ directory")
            sys.exit(1)
            
        print(f"Found {len(notebooks)} notebooks to validate")
        
        failed = 0
        for notebook in sorted(notebooks):
            if not validator.validate_notebook(notebook):
                failed += 1
                
        print(f"\n{'='*60}")
        print(f"Summary: {len(notebooks)-failed}/{len(notebooks)} notebooks passed validation")
        
        if failed > 0:
            sys.exit(1)
            
    elif args.notebook:
        notebook_path = Path(args.notebook)
        if not validator.validate_notebook(notebook_path):
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
        

if __name__ == '__main__':
    main()