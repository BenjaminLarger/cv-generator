"""
PDFGenerator for converting HTML content to professional PDF documents.

This module provides comprehensive PDF generation functionality using Playwright
with optimized settings for CV and cover letter documents, including proper
formatting, file organization, and error handling.
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

try:
    from playwright.async_api import async_playwright, Browser, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.job_data import JobData
from models.match_result import MatchResult
from utils.logging_config import get_agents_logger
from utils.file_manager import create_application_folder, generate_filename

logger = get_agents_logger()


class PDFGeneratorError(Exception):
    """Base exception for PDF generator errors."""
    pass


class PlaywrightNotAvailableError(PDFGeneratorError):
    """Exception raised when Playwright is not available."""
    pass


class BrowserInitializationError(PDFGeneratorError):
    """Exception raised when browser initialization fails."""
    pass


class PDFGenerationError(PDFGeneratorError):
    """Exception raised when PDF generation fails."""
    pass


class ValidationError(PDFGeneratorError):
    """Exception raised when input validation fails."""
    pass


@dataclass
class PDFOptions:
    """Configuration options for PDF generation."""
    format: str = "A4"
    margin_top: str = "1cm"
    margin_bottom: str = "1cm"
    margin_left: str = "1cm"
    margin_right: str = "1cm"
    print_background: bool = True
    prefer_css_page_size: bool = True
    generate_tagged_pdf: bool = False
    timeout: int = 30000  # 30 seconds
    wait_for_selector: Optional[str] = None
    wait_for_timeout: int = 2000  # 2 seconds
    scale: float = 1.0


class PDFGenerator:
    """
    Generates professional PDF documents from HTML content using Playwright.

    This class provides comprehensive PDF generation capabilities with proper
    browser management, error handling, and file organization for CV and
    cover letter documents.
    """

    def __init__(self, options: Optional[PDFOptions] = None):
        """
        Initialize the PDFGenerator.

        Args:
            options: Configuration options for PDF generation

        Raises:
            PlaywrightNotAvailableError: If Playwright is not installed
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise PlaywrightNotAvailableError(
                "Playwright is not available. Install it with: pip install playwright && playwright install"
            )

        self.options = options or PDFOptions()
        self.logger = logger
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self.logger.info("PDFGenerator initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_browser()

    async def generate_pdfs(
        self,
        approved_html: Dict[str, str],
        job_data: JobData,
        output_folder: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Main entry point for PDF generation.

        Args:
            approved_html: Dictionary with 'cv' and 'cover_letter' HTML content
            job_data: JobData instance with job information
            output_folder: Optional output folder (defaults to auto-generated)

        Returns:
            Dictionary with file paths: {'cv': 'path/to/cv.pdf', 'cover_letter': 'path/to/cover_letter.pdf'}

        Raises:
            ValidationError: If input validation fails
            PDFGenerationError: If PDF generation fails
        """
        self.logger.info(f"Generating PDFs for {job_data.company_name}")

        try:
            # Validate inputs
            self._validate_inputs(approved_html, job_data)

            # Create output folder if not provided
            if output_folder is None:
                output_folder = create_application_folder(
                    job_data.company_name,
                    datetime.now().strftime('%d-%m-%Y')
                )

            # Ensure documents subfolder exists
            documents_folder = Path(output_folder) / 'documents'
            documents_folder.mkdir(exist_ok=True)

            # Initialize browser if not already done
            if self._browser is None:
                await self._initialize_browser()

            # Generate PDFs
            pdf_paths = {}

            for doc_type, html_content in approved_html.items():
                if html_content.strip():  # Only generate if content exists
                    # Generate filename with proper naming convention
                    if doc_type == 'cv':
                        filename = self._generate_pdf_filename(
                            'cv',
                            job_data.company_name,
                            datetime.now().strftime('%d-%m-%Y')
                        )
                    else:
                        filename = self._generate_pdf_filename(
                            'cover_letter',
                            job_data.company_name,
                            datetime.now().strftime('%d-%m-%Y')
                        )

                    output_path = str(documents_folder / filename)

                    # Generate PDF
                    success = await self.html_to_pdf(html_content, output_path)

                    if success:
                        # Validate the generated PDF
                        if self.validate_pdf_output(output_path):
                            pdf_paths[doc_type] = output_path
                            self.logger.info(f"PDF generated successfully: {output_path}")
                        else:
                            self.logger.warning(f"PDF validation failed: {output_path}")
                            pdf_paths[doc_type] = output_path  # Still return path but log warning
                    else:
                        error_msg = f"Failed to generate PDF for {doc_type}"
                        self.logger.error(error_msg)
                        raise PDFGenerationError(error_msg)

            self.logger.info(f"All PDFs generated successfully. Total: {len(pdf_paths)}")
            return pdf_paths

        except Exception as e:
            error_msg = f"Failed to generate PDFs: {e}"
            self.logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

    async def html_to_pdf(self, html_content: str, output_path: str) -> bool:
        """
        Converts HTML content to PDF using Playwright with single-page constraint.

        Based on the reference ConverterHtmlToPdf model with improvements for
        single-page CV generation and better formatting control.

        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved

        Returns:
            Success status (True if successful, False otherwise)

        Raises:
            PDFGenerationError: If PDF generation fails
        """
        self.logger.debug(f"Converting HTML to PDF: {output_path}")

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Use a fresh browser instance for each conversion to avoid state issues
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )

                try:
                    page = await browser.new_page()

                    await page.set_content(html_content, wait_until='networkidle', timeout=self.options.timeout)

                    await page.wait_for_timeout(self.options.wait_for_timeout)  
                    
                    # Generate PDF with optimized settings for single page
                    await page.pdf(
                        path=output_path,
                        format="A4",
                        print_background=True,
                        prefer_css_page_size=False,  # Use our A4 format
                        margin={
                            'top': '1cm',
                            'bottom': '1cm',
                            'left': '1cm',
                            'right': '1cm'
                        },
                    )

                    self.logger.info(f"PDF generated successfully: {output_path}")
                    return True

                finally:
                    await browser.close()

        except Exception as e:
            error_msg = f"Error generating PDF: {str(e)}"
            self.logger.error(error_msg)
            raise PDFGenerationError(error_msg) from e

        # Insert CSS right after the <head> tag or before </head>
        if '<head>' in html_content:
            html_content = html_content.replace('<head>', f'<head>\n{single_page_css}')
        elif '<html>' in html_content:
            html_content = html_content.replace('<html>', f'<html>\n<head>\n{single_page_css}\n</head>')
        else:
            html_content = f'<html><head>{single_page_css}</head><body>{html_content}</body></html>'

        return html_content

    def validate_pdf_output(self, pdf_path: str) -> bool:
        """
        Validates generated PDF files.

        Args:
            pdf_path: Path to the PDF file to validate

        Returns:
            Validation results (True if valid, False otherwise)
        """
        self.logger.debug(f"Validating PDF output: {pdf_path}")

        try:
            pdf_file = Path(pdf_path)

            # Check if file exists
            if not pdf_file.exists():
                self.logger.error(f"PDF file does not exist: {pdf_path}")
                return False

            # Check file size (should be > 1KB for valid PDFs)
            file_size = pdf_file.stat().st_size
            if file_size < 1024:  # 1KB
                self.logger.error(f"PDF file too small ({file_size} bytes): {pdf_path}")
                return False

            # Check if file is readable
            try:
                with open(pdf_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'%PDF-'):
                        self.logger.error(f"Invalid PDF header: {pdf_path}")
                        return False
            except Exception as e:
                self.logger.error(f"Cannot read PDF file: {e}")
                return False

            # Check maximum file size (warn if > 5MB)
            max_size = 5 * 1024 * 1024  # 5MB
            if file_size > max_size:
                self.logger.warning(f"PDF file is large ({file_size // (1024*1024)}MB): {pdf_path}")

            self.logger.debug(f"PDF validation successful: {pdf_path} ({file_size} bytes)")
            return True

        except Exception as e:
            self.logger.error(f"Error validating PDF: {e}")
            return False

    async def _initialize_browser(self) -> None:
        """Initialize Playwright browser."""
        try:
            self.logger.debug("Initializing Playwright browser")

            self._playwright = await async_playwright().start()

            # Launch browser with optimized settings for PDF generation
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--run-all-compositor-stages-before-draw',
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-backgrounding-occluded-windows'
                ]
            )

            self.logger.info("Browser initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize browser: {e}"
            self.logger.error(error_msg)
            raise BrowserInitializationError(error_msg) from e

    async def _cleanup_browser(self) -> None:
        """Clean up browser resources."""
        try:
            if self._browser:
                await self._browser.close()
                self._browser = None
                self.logger.debug("Browser closed")

            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
                self.logger.debug("Playwright stopped")

        except Exception as e:
            self.logger.warning(f"Error during browser cleanup: {e}")

    def _validate_inputs(self, approved_html: Dict[str, str], job_data: JobData) -> None:
        """Validate input parameters."""
        if not isinstance(approved_html, dict):
            raise ValidationError("approved_html must be a dictionary")

        if not isinstance(job_data, JobData):
            raise ValidationError("job_data must be a JobData instance")

        required_keys = ['cv', 'cover_letter']
        for key in required_keys:
            if key not in approved_html:
                raise ValidationError(f"Missing required key in approved_html: {key}")

        # Check if at least one document has content
        has_content = any(content.strip() for content in approved_html.values())
        if not has_content:
            raise ValidationError("At least one document must have content")

    def _generate_pdf_filename(self, doc_type: str, company: str, date: str) -> str:
        """
        Generate PDF filename with proper naming convention.

        Args:
            doc_type: Type of document (cv, cover_letter)
            company: Company name
            date: Date in DD-MM-YYYY format

        Returns:
            Generated filename following the pattern: {doc_type}_{company}_{DD-MM-YYYY}.pdf
        """
        # Sanitize company name for filename
        sanitized_company = re.sub(r'[<>:"/\\|?*]', '', company)
        sanitized_company = re.sub(r'[^\w\s-]', '', sanitized_company)
        sanitized_company = re.sub(r'\s+', '_', sanitized_company.strip())
        sanitized_company = re.sub(r'_+', '_', sanitized_company)
        sanitized_company = sanitized_company.strip('_').lower()

        if not sanitized_company:
            sanitized_company = "company"

        # Ensure date is in DD-MM-YYYY format
        try:
            # Parse and reformat date to ensure consistency
            if '-' in date:
                parts = date.split('-')
                if len(parts) == 3:
                    if len(parts[0]) == 4:  # YYYY-MM-DD format
                        date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    # Already in DD-MM-YYYY format
        except Exception:
            # Fallback to current date
            date = datetime.now().strftime('%d-%m-%Y')

        return f"{doc_type}_{sanitized_company}_{date}.pdf"

    def _generate_pdf_css(self) -> str:
        """Generate PDF-optimized CSS styles."""
        return """
        @page {
            size: A4;
            margin: 1cm;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #000;
            background: white;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }

        .pdf-container {
            max-width: 100%;
            margin: 0;
            padding: 0;
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: bold;
            margin-bottom: 8pt;
            margin-top: 12pt;
            page-break-after: avoid;
        }

        h1 {
            font-size: 16pt;
            margin-bottom: 12pt;
        }

        h2 {
            font-size: 14pt;
            margin-bottom: 10pt;
        }

        h3 {
            font-size: 12pt;
            margin-bottom: 8pt;
        }

        p {
            margin-bottom: 6pt;
            text-align: justify;
            orphans: 2;
            widows: 2;
        }

        ul, ol {
            margin-bottom: 8pt;
            padding-left: 20pt;
        }

        li {
            margin-bottom: 2pt;
        }

        .section {
            margin-bottom: 16pt;
            page-break-inside: avoid;
        }

        .header {
            text-align: center;
            margin-bottom: 20pt;
            border-bottom: 1pt solid #000;
            padding-bottom: 8pt;
        }

        .contact-info {
            font-size: 10pt;
            margin-top: 6pt;
        }

        .skills-list, .experience-list {
            list-style-type: disc;
        }

        .date-range {
            font-style: italic;
            font-size: 10pt;
        }

        .company-name {
            font-weight: bold;
        }

        .job-title {
            font-weight: bold;
            margin-bottom: 2pt;
        }

        .achievements {
            margin-top: 4pt;
        }

        /* Ensure good page breaks */
        .experience-item, .education-item, .project-item {
            page-break-inside: avoid;
            margin-bottom: 12pt;
        }

        /* Remove any background colors for PDF */
        * {
            background: transparent !important;
        }

        /* Ensure text is black for readability */
        p, li, span, div {
            color: #000 !important;
        }

        /* Hide interactive elements */
        button, input, .interactive {
            display: none !important;
        }

        /* Optimize table layouts */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 8pt;
        }

        td, th {
            padding: 4pt;
            border: 1pt solid #000;
            vertical-align: top;
        }

        /* Special formatting for cover letters */
        .letter-header {
            margin-bottom: 20pt;
        }

        .letter-date {
            text-align: right;
            margin-bottom: 16pt;
        }

        .letter-address {
            margin-bottom: 16pt;
        }

        .letter-greeting {
            margin-bottom: 12pt;
        }

        .letter-body {
            text-align: justify;
            margin-bottom: 12pt;
        }

        .letter-closing {
            margin-top: 20pt;
        }

        .signature-space {
            margin-top: 40pt;
            margin-bottom: 8pt;
        }
        """

    # Synchronous wrapper methods for easier use
    def generate_pdfs_sync(
        self,
        approved_html: Dict[str, str],
        job_data: JobData,
        output_folder: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Synchronous wrapper for generate_pdfs.

        Args:
            approved_html: Dictionary with HTML content
            job_data: JobData instance
            output_folder: Optional output folder

        Returns:
            Dictionary with generated PDF paths
        """
        return asyncio.run(self._generate_pdfs_with_context(approved_html, job_data, output_folder))

    def html_to_pdf_sync(self, html_content: str, output_path: str) -> bool:
        """
        Synchronous wrapper for html_to_pdf.

        Args:
            html_content: HTML content to convert
            output_path: Output PDF path

        Returns:
            Success status
        """
        return asyncio.run(self._html_to_pdf_with_context(html_content, output_path))

    async def _generate_pdfs_with_context(
        self,
        approved_html: Dict[str, str],
        job_data: JobData,
        output_folder: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate PDFs with automatic context management."""
        async with self:
            return await self.generate_pdfs(approved_html, job_data, output_folder)

    async def _html_to_pdf_with_context(self, html_content: str, output_path: str) -> bool:
        """Generate single PDF with automatic context management."""
        async with self:
            return await self.html_to_pdf(html_content, output_path)


# Convenience function for quick PDF generation
async def generate_application_pdfs(
    cv_html: str,
    cover_letter_html: str,
    job_data: JobData,
    output_folder: Optional[str] = None,
    options: Optional[PDFOptions] = None
) -> Dict[str, str]:
    """
    Convenience function to generate both CV and cover letter PDFs.

    Args:
        cv_html: CV HTML content
        cover_letter_html: Cover letter HTML content
        job_data: JobData instance
        output_folder: Optional output folder
        options: Optional PDF generation options

    Returns:
        Dictionary with generated PDF paths

    Raises:
        PDFGenerationError: If PDF generation fails
    """
    approved_html = {
        'cv': cv_html,
        'cover_letter': cover_letter_html
    }

    async with PDFGenerator(options) as generator:
        return await generator.generate_pdfs(approved_html, job_data, output_folder)


def generate_application_pdfs_sync(
    cv_html: str,
    cover_letter_html: str,
    job_data: JobData,
    output_folder: Optional[str] = None,
    options: Optional[PDFOptions] = None
) -> Dict[str, str]:
    """
    Synchronous convenience function to generate both CV and cover letter PDFs.

    Args:
        cv_html: CV HTML content
        cover_letter_html: Cover letter HTML content
        job_data: JobData instance
        output_folder: Optional output folder
        options: Optional PDF generation options

    Returns:
        Dictionary with generated PDF paths

    Raises:
        PDFGenerationError: If PDF generation fails
    """
    return asyncio.run(generate_application_pdfs(
        cv_html, cover_letter_html, job_data, output_folder, options
    ))