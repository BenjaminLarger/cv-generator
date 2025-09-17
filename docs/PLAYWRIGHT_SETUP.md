# Playwright Setup Guide for PDF Generation

This guide provides comprehensive instructions for setting up Playwright for PDF generation in the CV/Cover Letter Generator project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Browser Setup](#browser-setup)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Advanced Configuration](#advanced-configuration)

## Prerequisites

Before installing Playwright, ensure you have:

- Python 3.8 or higher
- pip package manager
- Sufficient disk space (browsers require ~1GB total)
- Internet connection for browser downloads
- Administrative privileges (for some system-level dependencies)

### System Requirements

| OS | Requirements |
|---|---|
| **Linux** | glibc 2.28+, xvfb (for headless mode), fonts |
| **macOS** | macOS 11.0+ |
| **Windows** | Windows 10 version 1903+ |

## Installation

### Step 1: Install Playwright Python Package

Install Playwright using pip or uv (recommended):

```bash
# Using uv (recommended)
uv add playwright

# Or using pip
pip install playwright
```

### Step 2: Install Browser Binaries

After installing the Python package, install the browser binaries:

```bash
# Install all browsers (recommended for development)
playwright install

# Or install only Chromium (sufficient for PDF generation)
playwright install chromium

# Install system dependencies (Linux only)
playwright install-deps
```

### Step 3: Verify Installation

Create a test script to verify the installation:

```python
# test_playwright.py
import asyncio
from playwright.async_api import async_playwright

async def test_playwright():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('https://example.com')
        title = await page.title()
        print(f"Page title: {title}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_playwright())
```

Run the test:
```bash
python test_playwright.py
```

## Browser Setup

### Chromium Configuration

The PDF generator uses Chromium with optimized settings for document generation:

```python
from playwright.async_api import async_playwright

# Optimized browser launch options for PDF generation
browser_args = [
    '--no-sandbox',                    # Disable sandbox for headless environments
    '--disable-setuid-sandbox',        # Additional security bypass for containers
    '--disable-dev-shm-usage',         # Use /tmp instead of /dev/shm
    '--disable-gpu',                   # Disable GPU acceleration
    '--disable-web-security',          # Allow local file access
    '--disable-features=VizDisplayCompositor',
    '--run-all-compositor-stages-before-draw',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows'
]

async with async_playwright() as p:
    browser = await p.chromium.launch(
        headless=True,
        args=browser_args
    )
```

### Font Configuration

For consistent PDF rendering across systems, ensure proper font availability:

#### Linux Font Setup
```bash
# Install common fonts
sudo apt-get update
sudo apt-get install -y fonts-liberation fonts-dejavu-core fonts-freefont-ttf

# For better compatibility with Windows fonts
sudo apt-get install -y ttf-mscorefonts-installer

# Install Chromium dependencies
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libgbm1
```

#### macOS Font Setup
```bash
# Fonts are usually available by default
# If needed, install additional fonts via Homebrew
brew install font-liberation
```

#### Windows Font Setup
Windows typically has all required fonts pre-installed.

## Configuration

### Environment Variables

Configure Playwright behavior using environment variables:

```bash
# Disable browser downloads if browsers are pre-installed
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# Custom browser download path
export PLAYWRIGHT_BROWSERS_PATH="/custom/path/to/browsers"

# Disable browser updates
export PLAYWRIGHT_SKIP_BROWSER_GC=1

# Enable debug logging
export DEBUG=pw:*
```

### Docker Configuration

For containerized deployments:

```dockerfile
# Dockerfile excerpt for Playwright support
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium && \
    playwright install-deps chromium

# Set environment variables
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
```

### PDF Generation Settings

Configure PDF generation options in your application:

```python
from src.agents.pdf_generator import PDFOptions

# Standard configuration
pdf_options = PDFOptions(
    format="A4",
    margin_top="1cm",
    margin_bottom="1cm",
    margin_left="1cm",
    margin_right="1cm",
    print_background=True,
    prefer_css_page_size=True,
    timeout=30000,  # 30 seconds
    scale=1.0
)

# High-quality configuration
high_quality_options = PDFOptions(
    format="A4",
    margin_top="1.5cm",
    margin_bottom="1.5cm",
    margin_left="2cm",
    margin_right="2cm",
    print_background=True,
    prefer_css_page_size=True,
    generate_tagged_pdf=True,  # For accessibility
    timeout=45000,  # 45 seconds
    scale=0.9,  # Slightly smaller for better fit
    wait_for_timeout=3000  # Wait longer for rendering
)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Browser Installation Fails
```bash
# Error: Failed to download browsers
# Solution: Check internet connection and disk space
df -h  # Check disk space
playwright install --verbose  # Verbose output for debugging
```

#### Issue 2: Permission Denied Errors
```bash
# Error: Permission denied accessing browser
# Solution: Check file permissions and ownership
ls -la $(playwright browsers-path)
sudo chown -R $(whoami) $(playwright browsers-path)
```

#### Issue 3: Headless Mode Issues on Linux
```bash
# Error: Browser crashes in headless mode
# Solution: Install xvfb and additional dependencies
sudo apt-get install -y xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

#### Issue 4: Font Rendering Issues
```bash
# Error: Fonts not rendering correctly in PDFs
# Solution: Install additional font packages
sudo apt-get install -y fontconfig fonts-dejavu-core
fc-cache -fv  # Rebuild font cache
```

#### Issue 5: Timeout Errors
```python
# Error: Page timeout during PDF generation
# Solution: Increase timeout and add wait conditions
pdf_options = PDFOptions(
    timeout=60000,  # Increase to 60 seconds
    wait_for_timeout=5000,  # Wait 5 seconds for rendering
    wait_for_selector="body"  # Wait for specific element
)
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
from src.utils.logging_config import get_agents_logger

# Enable debug logging
logger = get_agents_logger()
logger.setLevel(logging.DEBUG)

# Or set environment variable
import os
os.environ['DEBUG'] = 'pw:*'
```

### Memory Issues

For memory-intensive operations:

```python
# Optimize memory usage
async with PDFGenerator() as generator:
    # Process documents one at a time
    for doc_type, html_content in approved_html.items():
        pdf_path = await generator.html_to_pdf(html_content, f"{doc_type}.pdf")
        # Process immediately, don't accumulate
```

## Best Practices

### 1. Resource Management
```python
# Always use context managers
async with PDFGenerator() as generator:
    # PDF generation code here
    pass
# Browser automatically cleaned up
```

### 2. Error Handling
```python
from src.agents.pdf_generator import PDFGenerationError

try:
    pdf_paths = await generator.generate_pdfs(html_content, job_data)
except PDFGenerationError as e:
    logger.error(f"PDF generation failed: {e}")
    # Implement fallback or retry logic
```

### 3. Performance Optimization
```python
# Reuse browser instances for multiple PDFs
async with PDFGenerator() as generator:
    for application in applications:
        pdf_paths = await generator.generate_pdfs(
            application.html,
            application.job_data
        )
```

### 4. Security Considerations
```python
# Sanitize HTML content before PDF generation
from html import escape

def sanitize_for_pdf(html_content):
    # Remove potentially dangerous scripts
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
    # Escape user-provided content
    return html_content
```

## Advanced Configuration

### Custom Browser Configuration
```python
class CustomPDFGenerator(PDFGenerator):
    async def _initialize_browser(self):
        self._playwright = await async_playwright().start()

        # Custom browser configuration
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--memory-pressure-off',
                '--max_old_space_size=4096',  # Increase memory limit
                '--js-flags="--max-old-space-size=4096"'
            ],
            env={
                'FONTCONFIG_FILE': '/path/to/custom/fonts.conf'
            }
        )
```

### Custom CSS for PDF
```python
def apply_custom_pdf_styling(self, html: str) -> str:
    custom_css = """
    @page {
        size: A4;
        margin: 2cm;
        @top-center {
            content: "Confidential - " attr(title);
        }
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
        }
    }

    .page-break {
        page-break-before: always;
    }

    .no-break {
        page-break-inside: avoid;
    }
    """

    return self._inject_css(html, custom_css)
```

### Batch Processing
```python
async def batch_generate_pdfs(applications: List[Dict], max_concurrent: int = 3):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_single(app_data):
        async with semaphore:
            async with PDFGenerator() as generator:
                return await generator.generate_pdfs(
                    app_data['html'],
                    app_data['job_data']
                )

    tasks = [generate_single(app) for app in applications]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results
```

## Testing

### Unit Tests
```python
import pytest
from src.agents.pdf_generator import PDFGenerator, PDFOptions

@pytest.mark.asyncio
async def test_pdf_generation():
    html_content = "<html><body><h1>Test</h1></body></html>"

    async with PDFGenerator() as generator:
        result = await generator.html_to_pdf(html_content, "test.pdf")
        assert result is True
        assert generator.validate_pdf_output("test.pdf") is True
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_complete_workflow():
    job_data = create_sample_job_data()
    html_content = create_sample_html_content()

    async with PDFGenerator() as generator:
        pdf_paths = await generator.generate_pdfs(html_content, job_data)

        assert 'cv' in pdf_paths
        assert 'cover_letter' in pdf_paths

        for path in pdf_paths.values():
            assert Path(path).exists()
            assert generator.validate_pdf_output(path)
```

## Performance Monitoring

Monitor PDF generation performance:

```python
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None

    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss

    def end(self):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        return {
            'duration': end_time - self.start_time,
            'memory_delta': end_memory - self.start_memory,
            'peak_memory': max(self.start_memory, end_memory)
        }

# Usage
monitor = PerformanceMonitor()
monitor.start()
pdf_paths = await generator.generate_pdfs(html_content, job_data)
stats = monitor.end()
logger.info(f"PDF generation took {stats['duration']:.2f}s, used {stats['memory_delta']/1024/1024:.1f}MB")
```

This setup guide provides everything needed to successfully configure and use Playwright for PDF generation in the CV/Cover Letter Generator project.