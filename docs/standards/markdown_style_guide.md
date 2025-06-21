# Markdown Style Guide

## Purpose
This guide establishes consistent documentation standards for the Vision Transformer Network Traffic Analysis project.

## General Principles
- Write clear, concise documentation
- Use active voice
- Include code examples for every concept
- Build complexity progressively
- Test all code examples before including

## Markdown Formatting Standards

### Headers
- Use ATX-style headers (# symbols)
- Maintain logical hierarchy (don't skip levels)
- Include one blank line before and after headers
- Capitalize headers using title case

```markdown
# Main Title

## Section Title

### Subsection Title
```

### Lists
- Use `-` for unordered lists
- Use `1.` for ordered lists
- Indent nested lists with 2 spaces
- Include blank line before and after lists

### Code Blocks
- Use triple backticks with language specification
- Include descriptive comments in code
- Keep examples concise and focused
- Show expected output when relevant

```python
# Example: Converting packet bytes to image
import numpy as np

def bytes_to_image(packet_bytes, image_size=224):
    """Convert packet bytes to grayscale image."""
    # Pad or truncate to fit image dimensions
    total_pixels = image_size * image_size
    if len(packet_bytes) < total_pixels:
        packet_bytes = np.pad(packet_bytes, (0, total_pixels - len(packet_bytes)))
    else:
        packet_bytes = packet_bytes[:total_pixels]
    
    # Reshape to 2D image
    return packet_bytes.reshape(image_size, image_size)
```

### Links and References
- Use descriptive link text
- Prefer reference-style links for repeated URLs
- Include tooltips for technical terms

```markdown
[Vision Transformer paper][vit-paper] introduces the architecture.

[vit-paper]: https://arxiv.org/abs/2010.11929 "An Image is Worth 16x16 Words"
```

### Tables
- Use pipes and hyphens for tables
- Align columns for readability
- Include header row

```markdown
| Encoding Method | Pros | Cons |
|-----------------|------|------|
| Sequential | Simple, preserves order | May miss spatial patterns |
| Hilbert Curve | Better locality | More complex |
| Spiral | Good for centered data | Implementation overhead |
```

### Emphasis
- Use `**bold**` for strong emphasis
- Use `*italic*` for subtle emphasis
- Use `***bold italic***` sparingly
- Use `` `inline code` `` for technical terms

## Document Structure

### Front Matter
Every document should begin with:
1. Title (H1)
2. Brief description/purpose
3. Table of contents (for documents > 3 sections)
4. Prerequisites (if applicable)

### Body Content
- Start with overview/context
- Progress from simple to complex
- Include examples after each concept
- Add visualizations where helpful

### End Matter
- Summary of key points
- Links to related documents
- References/citations
- Last updated date

## Writing Style

### Tone
- Professional but approachable
- Encourage learning
- Acknowledge complexity
- Provide clear guidance

### Technical Accuracy
- Define acronyms on first use
- Explain assumptions
- Include version numbers
- Specify hardware requirements

### Examples

#### Good Example
```markdown
## Understanding Attention Mechanisms

The attention mechanism allows the model to focus on different parts of the input when making predictions. In our context, this means the ViT can identify which packet bytes are most relevant for detecting malware.

**Key concept**: Attention weights indicate the importance of each patch relative to others.

Here's a simplified example:
```

#### Poor Example
```markdown
## Attention

Attention is when the model looks at stuff. It's important for finding bad packets.
```

## Notebook Documentation Standards

### Cell Structure
1. Markdown explanation
2. Code implementation
3. Output/visualization
4. Interpretation

### Markdown Cells
- Explain the "why" before the "how"
- Include learning objectives
- Add tips and warnings
- Reference external resources

### Code Cells
- Import statements at the top
- Define constants clearly
- Use descriptive variable names
- Include inline comments for complex logic

## File Naming Conventions

### Documents
- Use lowercase with hyphens: `packet-encoding-guide.md`
- Be descriptive but concise
- Include version in filename if needed: `api-reference-v2.md`

### Notebooks
- Follow numbered sequence: `01_data_exploration.ipynb`
- Use descriptive suffixes: `03_vit_implementation_basic.ipynb`
- Indicate difficulty: `07_few_shot_advanced.ipynb`

## Review Checklist

Before finalizing any documentation:
- [ ] Spell check and grammar review
- [ ] All code examples tested
- [ ] Links verified
- [ ] Formatting consistent
- [ ] Technical accuracy verified
- [ ] Accessible to target audience
- [ ] Includes practical examples
- [ ] Cross-references updated

## Version Control

### Commit Messages
- Use present tense: "Add encoding tutorial"
- Be specific: "Fix typo in Hilbert curve example"
- Reference issues: "Update API docs for #123"

### Documentation Updates
- Update last modified date
- Note significant changes
- Maintain changelog for major documents