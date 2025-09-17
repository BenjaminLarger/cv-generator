# Preview and PDF Generation System

A comprehensive system for creating interactive HTML previews and generating professional PDF documents for CV and cover letter applications.

## Overview

The Preview and PDF Generation System provides two main components:

1. **PreviewGenerator** - Creates interactive HTML previews with side-by-side comparisons, match analysis, and approval interfaces
2. **PDFGenerator** - Converts HTML content to professional PDF documents using Playwright with optimized settings

## Features

### PreviewGenerator Features
- 📊 **Interactive Match Analysis** - Visual match score display with detailed breakdowns
- 🔍 **Side-by-Side Comparison** - Original vs customized content comparison
- ✨ **Change Highlighting** - Clear visualization of customizations made
- 👤 **Approval Interface** - Interactive buttons for approve/reject workflow
- 🎨 **Professional Styling** - Clean, responsive design with multiple themes
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile devices
- 🖨️ **Print Optimization** - Print-friendly styles for preview documents

### PDFGenerator Features
- 🚀 **Playwright Integration** - High-quality PDF generation using Chromium
- 📄 **Professional Formatting** - Optimized for ATS compatibility and print quality
- 📁 **Organized File Structure** - Systematic file naming and folder organization
- ⚡ **Async Support** - Efficient async/await patterns for performance
- 🔧 **Configurable Options** - Extensive customization of PDF generation settings
- ✅ **Validation** - Comprehensive PDF validation and error handling
- 🏷️ **Naming Convention** - `cv_{company}_{DD-MM-YYYY}.pdf` format
- 🔄 **Retry Logic** - Automatic retry for transient failures

## Quick Start

### Prerequisites
```bash
# Install required dependencies
uv add playwright
playwright install chromium

# Or using pip
pip install playwright
playwright install chromium
```

### Basic Usage

```python
from src.agents.preview_generator import PreviewGenerator
from src.agents.pdf_generator import PDFGenerator, generate_application_pdfs_sync
from src.models.job_data import JobData
from src.models.match_result import MatchResult

# Create preview
preview_generator = PreviewGenerator()
preview_html = preview_generator.generate_preview(
    customized_html={'cv': cv_html, 'cover_letter': cover_letter_html},
    match_result=match_result,
    changes=changes_list,
    original_html=original_content  # Optional
)

# Generate PDFs
pdf_paths = generate_application_pdfs_sync(
    cv_html=cv_html,
    cover_letter_html=cover_letter_html,
    job_data=job_data
)
```

## Detailed Documentation

### PreviewGenerator

#### Initialization
```python
from src.agents.preview_generator import PreviewGenerator, PreviewOptions

# Basic initialization
generator = PreviewGenerator()

# With custom options
options = PreviewOptions(
    show_side_by_side=True,
    include_match_analysis=True,
    enable_interactive_approval=True,
    highlight_changes=True,
    theme="professional",  # professional, modern, minimal
    max_content_length=10000
)
generator = PreviewGenerator(options)
```

#### Main Methods

##### `generate_preview()`
Main entry point for creating interactive previews.

```python
preview_html = generator.generate_preview(
    customized_html: Dict[str, str],      # Required: {'cv': html, 'cover_letter': html}
    match_result: MatchResult,            # Required: Match analysis results
    changes: List[str],                   # Required: List of changes made
    original_html: Optional[Dict[str, str]] = None  # Optional: For comparison
) -> str
```

**Parameters:**
- `customized_html`: Dictionary with 'cv' and 'cover_letter' HTML content
- `match_result`: MatchResult instance with job matching analysis
- `changes`: List of strings describing customizations made
- `original_html`: Optional original content for side-by-side comparison

**Returns:** Complete HTML string ready for browser display

##### `create_side_by_side_comparison()`
Creates visual comparison between original and customized content.

```python
comparison_html = generator.create_side_by_side_comparison(
    original: Dict[str, str],
    customized: Dict[str, str]
) -> str
```

##### `create_preview_html()`
Combines individual components into unified preview.

```python
preview_html = generator.create_preview_html(
    cv_html: str,
    cover_letter_html: str,
    match_data: MatchResult,
    comparison_html: str = "",
    changes: Optional[List[str]] = None
) -> str
```

#### Preview Components

The generated preview includes:

1. **Header Section**
   - Job title and company name
   - Generation timestamp
   - Professional styling

2. **Match Analysis Section**
   - Circular progress indicator for match score
   - Detailed statistics breakdown
   - Match category (Excellent, Good, Average, etc.)
   - Skills and experience percentages

3. **Changes Summary**
   - Bulleted list of customizations
   - Count of total changes
   - Visual highlighting

4. **Comparison Section** (optional)
   - Side-by-side original vs customized
   - Color-coded highlighting
   - Separate sections for CV and cover letter

5. **Document Display**
   - Clean, professional rendering
   - Scrollable content areas
   - Print-optimized styling

6. **Approval Interface**
   - Approve/Reject buttons
   - Download preview option
   - Status messages and feedback

### PDFGenerator

#### Initialization
```python
from src.agents.pdf_generator import PDFGenerator, PDFOptions

# Basic initialization
generator = PDFGenerator()

# With custom options
options = PDFOptions(
    format="A4",
    margin_top="1.5cm",
    margin_bottom="1.5cm",
    margin_left="2cm",
    margin_right="2cm",
    print_background=True,
    prefer_css_page_size=True,
    generate_tagged_pdf=True,  # For accessibility
    timeout=30000,  # 30 seconds
    wait_for_timeout=2000,  # 2 seconds
    scale=0.9
)
generator = PDFGenerator(options)
```

#### Main Methods

##### `generate_pdfs()` (Async)
Main entry point for PDF generation.

```python
async with PDFGenerator(options) as generator:
    pdf_paths = await generator.generate_pdfs(
        approved_html: Dict[str, str],      # Required: {'cv': html, 'cover_letter': html}
        job_data: JobData,                  # Required: Job information
        output_folder: Optional[str] = None # Optional: Custom output folder
    ) -> Dict[str, str]
```

**Returns:** Dictionary with file paths: `{'cv': 'path/to/cv.pdf', 'cover_letter': 'path/to/cover_letter.pdf'}`

##### `html_to_pdf()` (Async)
Converts individual HTML content to PDF.

```python
async with PDFGenerator() as generator:
    success = await generator.html_to_pdf(
        html_content: str,
        output_path: str
    ) -> bool
```

##### `apply_pdf_styling()`
Adds PDF-optimized CSS styling.

```python
styled_html = generator.apply_pdf_styling(html_content)
```

##### `validate_pdf_output()`
Validates generated PDF files.

```python
is_valid = generator.validate_pdf_output(pdf_path)
```

#### Synchronous Wrappers

For easier integration, synchronous wrapper functions are provided:

```python
# Synchronous PDF generation
pdf_paths = generator.generate_pdfs_sync(approved_html, job_data)

# Synchronous single PDF
success = generator.html_to_pdf_sync(html_content, output_path)

# Convenience function
pdf_paths = generate_application_pdfs_sync(cv_html, cover_letter_html, job_data)
```

#### File Organization

Generated files follow this structure:
```
applications/
└── company_name_YYYY-MM-DD/
    ├── documents/
    │   ├── cv_company_name_DD-MM-YYYY.pdf
    │   └── cover_letter_company_name_DD-MM-YYYY.pdf
    ├── analysis/
    ├── templates/
    ├── resources/
    └── metadata.json
```

## Configuration

### Preview Options

```python
@dataclass
class PreviewOptions:
    show_side_by_side: bool = True          # Enable comparison view
    include_match_analysis: bool = True      # Show match analysis
    enable_interactive_approval: bool = True # Show approval buttons
    highlight_changes: bool = True           # Highlight customizations
    theme: str = "professional"             # Theme: professional, modern, minimal
    max_content_length: int = 10000         # Max content length validation
```

### PDF Options

```python
@dataclass
class PDFOptions:
    format: str = "A4"                      # Page format
    margin_top: str = "1cm"                 # Top margin
    margin_bottom: str = "1cm"              # Bottom margin
    margin_left: str = "1cm"                # Left margin
    margin_right: str = "1cm"               # Right margin
    print_background: bool = True           # Include backgrounds
    prefer_css_page_size: bool = True       # Use CSS page size
    generate_tagged_pdf: bool = False       # Accessibility tags
    timeout: int = 30000                    # Generation timeout (ms)
    wait_for_selector: Optional[str] = None # Wait for specific element
    wait_for_timeout: int = 2000           # General wait time (ms)
    scale: float = 1.0                     # Content scale factor
```

## Error Handling

### Common Exceptions

```python
from src.agents.preview_generator import (
    PreviewGeneratorError,
    HTMLGenerationError,
    ValidationError
)

from src.agents.pdf_generator import (
    PDFGeneratorError,
    PlaywrightNotAvailableError,
    BrowserInitializationError,
    PDFGenerationError
)

# Example error handling
try:
    preview_html = generator.generate_preview(customized_html, match_result, changes)
except ValidationError as e:
    logger.error(f"Input validation failed: {e}")
except HTMLGenerationError as e:
    logger.error(f"HTML generation failed: {e}")
```

### Validation

Both generators include comprehensive input validation:

- **PreviewGenerator**: Validates HTML content, match results, and configuration
- **PDFGenerator**: Validates HTML content, job data, and PDF output

### Retry Logic

PDF generation includes automatic retry for transient failures:

```python
# Built-in retry for browser initialization
# Timeout handling for long-running operations
# Validation of generated files
```

## Examples

### Complete Workflow Example

```python
import asyncio
from src.agents.preview_generator import PreviewGenerator
from src.agents.pdf_generator import PDFGenerator

async def complete_workflow():
    # Step 1: Generate preview
    preview_generator = PreviewGenerator()
    preview_html = preview_generator.generate_preview(
        customized_html=customized_content,
        match_result=match_analysis,
        changes=customization_list,
        original_html=original_content
    )

    # Save preview for user review
    with open('preview.html', 'w') as f:
        f.write(preview_html)

    # Step 2: User approval (simulated)
    user_approved = True  # In real app, this would be interactive

    if user_approved:
        # Step 3: Generate PDFs
        async with PDFGenerator() as pdf_gen:
            pdf_paths = await pdf_gen.generate_pdfs(
                approved_html=customized_content,
                job_data=job_information
            )

        print(f"PDFs generated: {pdf_paths}")

# Run the workflow
asyncio.run(complete_workflow())
```

### Batch Processing Example

```python
async def batch_generate_pdfs(applications):
    async with PDFGenerator() as generator:
        results = []
        for app in applications:
            try:
                pdf_paths = await generator.generate_pdfs(
                    app['html'],
                    app['job_data']
                )
                results.append({'success': True, 'paths': pdf_paths})
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        return results
```

### Custom Styling Example

```python
# Custom PDF options for different document types
cv_options = PDFOptions(
    format="A4",
    margin_top="2cm",
    margin_bottom="2cm",
    scale=0.95,
    generate_tagged_pdf=True
)

cover_letter_options = PDFOptions(
    format="A4",
    margin_top="2.5cm",
    margin_bottom="2.5cm",
    scale=0.9
)
```

## Testing

### Unit Tests

```python
import pytest
from src.agents.preview_generator import PreviewGenerator
from src.agents.pdf_generator import PDFGenerator

def test_preview_generation():
    generator = PreviewGenerator()
    html = generator.generate_preview(sample_html, sample_match, sample_changes)
    assert '<html' in html
    assert 'match-analysis' in html

@pytest.mark.asyncio
async def test_pdf_generation():
    async with PDFGenerator() as generator:
        success = await generator.html_to_pdf(sample_html, 'test.pdf')
        assert success is True
        assert generator.validate_pdf_output('test.pdf') is True
```

### Integration Tests

See `examples/preview_and_pdf_examples.py` for comprehensive testing examples.

## Performance Considerations

### Memory Management
- Use async context managers for browser resources
- Process documents individually for large batches
- Monitor memory usage during generation

### Optimization Tips
- Reuse browser instances when generating multiple PDFs
- Optimize HTML content before generation
- Use appropriate timeout values
- Implement caching for repeated operations

### Resource Cleanup
```python
# Always use context managers
async with PDFGenerator() as generator:
    # Your code here
    pass
# Resources automatically cleaned up
```

## Browser Setup

For detailed browser setup instructions, see [PLAYWRIGHT_SETUP.md](./PLAYWRIGHT_SETUP.md).

## Troubleshooting

### Common Issues

1. **Playwright Not Available**
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. **Permission Errors**
   ```bash
   sudo chown -R $(whoami) $(playwright browsers-path)
   ```

3. **Font Rendering Issues**
   ```bash
   sudo apt-get install -y fonts-liberation fonts-dejavu-core
   ```

4. **Timeout Errors**
   - Increase timeout values in PDFOptions
   - Check system resources and load

5. **Large File Sizes**
   - Optimize images in HTML content
   - Reduce scale factor in PDFOptions
   - Use print-specific CSS

## API Reference

### PreviewGenerator Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `generate_preview()` | Main preview generation | `customized_html`, `match_result`, `changes`, `original_html?` | `str` |
| `create_side_by_side_comparison()` | Comparison view | `original`, `customized` | `str` |
| `create_preview_html()` | Unified preview | `cv_html`, `cover_letter_html`, `match_data`, `comparison_html?`, `changes?` | `str` |
| `get_user_approval()` | User approval interface | None | `bool` |

### PDFGenerator Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `generate_pdfs()` | Main PDF generation | `approved_html`, `job_data`, `output_folder?` | `Dict[str, str]` |
| `html_to_pdf()` | Single PDF generation | `html_content`, `output_path` | `bool` |
| `apply_pdf_styling()` | Add PDF CSS | `html` | `str` |
| `validate_pdf_output()` | Validate PDF | `pdf_path` | `bool` |

## License

This system is part of the IntelligentApply CV Generator project.