"""
Test suite for the Preview and PDF Generation System.

This module provides comprehensive tests for both the PreviewGenerator
and PDFGenerator classes, covering various scenarios and edge cases.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.preview_generator import (
    PreviewGenerator, PreviewOptions, PreviewGeneratorError,
    HTMLGenerationError, ValidationError
)
from src.models.job_data import JobData, ExperienceLevel
from src.models.match_result import (
    MatchResult, MatchScore, AnalysisDetails, RelevantExperience,
    RecommendedProject, ImprovementSuggestion, SuggestionCategory
)

# Mock Playwright import to avoid dependency in testing
with patch.dict('sys.modules', {'playwright.async_api': Mock()}):
    from src.agents.pdf_generator import (
        PDFGenerator, PDFOptions, PDFGeneratorError,
        PlaywrightNotAvailableError, BrowserInitializationError,
        PDFGenerationError
    )


# Test Data Fixtures
@pytest.fixture
def sample_job_data():
    """Create sample job data for testing."""
    return JobData(
        company_name="Test Company",
        position="Software Developer",
        requirements=["Python", "Django", "AWS"],
        skills_required=["Python", "Django", "PostgreSQL"],
        experience_level=ExperienceLevel.MID,
        description="Test job description for software developer position."
    )


@pytest.fixture
def sample_match_result():
    """Create sample match result for testing."""
    analysis_details = AnalysisDetails(
        skills_match_percentage=75.0,
        experience_match_percentage=80.0,
        education_match=True,
        missing_skills=["AWS"],
        additional_skills=["React"],
        strength_areas=["Python Development"],
        weakness_areas=["Cloud Platforms"]
    )

    return MatchResult(
        score=MatchScore.GOOD,
        matched_skills=["Python", "Django"],
        relevant_experiences=[],
        recommended_projects=[],
        suggestions=[],
        analysis_details=analysis_details,
        job_title="Software Developer",
        company_name="Test Company",
        confidence_level=0.85
    )


@pytest.fixture
def sample_html_content():
    """Create sample HTML content for testing."""
    return {
        'cv': '<div class="cv"><h1>John Doe</h1><p>Software Developer</p></div>',
        'cover_letter': '<div class="letter"><p>Dear Hiring Manager,</p><p>I am interested in the position.</p></div>'
    }


@pytest.fixture
def sample_original_content():
    """Create sample original content for comparison."""
    return {
        'cv': '<div class="cv"><h1>John Doe</h1><p>Developer</p></div>',
        'cover_letter': '<div class="letter"><p>Dear Sir/Madam,</p><p>I want to apply.</p></div>'
    }


@pytest.fixture
def sample_changes():
    """Create sample changes list for testing."""
    return [
        "Enhanced job title to match position requirements",
        "Improved cover letter opening to be more specific",
        "Added relevant skills to CV"
    ]


class TestPreviewGenerator:
    """Test cases for PreviewGenerator class."""

    def test_preview_generator_initialization_default(self):
        """Test default initialization of PreviewGenerator."""
        generator = PreviewGenerator()
        assert generator.options.show_side_by_side is True
        assert generator.options.include_match_analysis is True
        assert generator.options.theme == "professional"

    def test_preview_generator_initialization_custom(self):
        """Test custom initialization of PreviewGenerator."""
        options = PreviewOptions(
            show_side_by_side=False,
            theme="modern",
            max_content_length=5000
        )
        generator = PreviewGenerator(options)
        assert generator.options.show_side_by_side is False
        assert generator.options.theme == "modern"
        assert generator.options.max_content_length == 5000

    def test_generate_preview_success(self, sample_html_content, sample_match_result, sample_changes):
        """Test successful preview generation."""
        generator = PreviewGenerator()

        preview_html = generator.generate_preview(
            customized_html=sample_html_content,
            match_result=sample_match_result,
            changes=sample_changes
        )

        # Check that the HTML contains expected elements
        assert '<html' in preview_html
        assert 'match-analysis' in preview_html
        assert 'changes-summary' in preview_html
        assert 'documents-section' in preview_html
        assert 'approval-interface' in preview_html
        assert sample_match_result.company_name in preview_html
        assert sample_match_result.job_title in preview_html

    def test_generate_preview_with_comparison(self, sample_html_content, sample_original_content,
                                            sample_match_result, sample_changes):
        """Test preview generation with side-by-side comparison."""
        generator = PreviewGenerator()

        preview_html = generator.generate_preview(
            customized_html=sample_html_content,
            match_result=sample_match_result,
            changes=sample_changes,
            original_html=sample_original_content
        )

        assert 'comparisons-section' in preview_html
        assert 'comparison-grid' in preview_html
        assert 'original-content' in preview_html
        assert 'customized-content' in preview_html

    def test_generate_preview_validation_errors(self, sample_match_result, sample_changes):
        """Test validation errors in preview generation."""
        generator = PreviewGenerator()

        # Test missing required keys
        with pytest.raises(ValidationError):
            generator.generate_preview(
                customized_html={'invalid_key': 'content'},
                match_result=sample_match_result,
                changes=sample_changes
            )

        # Test invalid match result type
        with pytest.raises(ValidationError):
            generator.generate_preview(
                customized_html={'cv': 'content', 'cover_letter': 'content'},
                match_result="invalid",
                changes=sample_changes
            )

        # Test invalid changes type
        with pytest.raises(ValidationError):
            generator.generate_preview(
                customized_html={'cv': 'content', 'cover_letter': 'content'},
                match_result=sample_match_result,
                changes="invalid"
            )

    def test_create_side_by_side_comparison(self, sample_html_content, sample_original_content):
        """Test side-by-side comparison creation."""
        generator = PreviewGenerator()

        comparison_html = generator.create_side_by_side_comparison(
            original=sample_original_content,
            customized=sample_html_content
        )

        assert 'comparison-section' in comparison_html
        assert 'CV Comparison' in comparison_html
        assert 'Cover Letter Comparison' in comparison_html

    def test_create_preview_html_components(self, sample_html_content, sample_match_result, sample_changes):
        """Test individual components of preview HTML."""
        generator = PreviewGenerator()

        preview_html = generator.create_preview_html(
            cv_html=sample_html_content['cv'],
            cover_letter_html=sample_html_content['cover_letter'],
            match_data=sample_match_result,
            changes=sample_changes
        )

        # Check for all major components
        assert 'preview-header' in preview_html
        assert 'match-analysis' in preview_html
        assert 'changes-summary' in preview_html
        assert 'documents-section' in preview_html
        assert 'cv-section' in preview_html
        assert 'cover-letter-section' in preview_html

    def test_html_sanitization(self):
        """Test HTML content sanitization."""
        generator = PreviewGenerator()

        # Test with malicious content
        malicious_html = '<script>alert("xss")</script><p>Safe content</p>'
        sanitized = generator._sanitize_html_content(malicious_html)

        # Should not contain script tags
        assert '<script>' not in sanitized
        assert 'Safe content' in sanitized

    def test_empty_content_handling(self, sample_match_result, sample_changes):
        """Test handling of empty content."""
        generator = PreviewGenerator()

        # Test with empty HTML content
        empty_html = {'cv': '', 'cover_letter': ''}

        preview_html = generator.generate_preview(
            customized_html=empty_html,
            match_result=sample_match_result,
            changes=sample_changes
        )

        assert 'No content available' in preview_html

    def test_get_user_approval(self):
        """Test user approval functionality."""
        generator = PreviewGenerator()

        # Currently returns True as placeholder
        approval = generator.get_user_approval()
        assert approval is True

    def test_content_length_validation(self, sample_match_result, sample_changes):
        """Test content length validation."""
        options = PreviewOptions(max_content_length=10)
        generator = PreviewGenerator(options)

        long_content = {'cv': 'x' * 100, 'cover_letter': 'y' * 100}

        with pytest.raises(ValidationError):
            generator.generate_preview(
                customized_html=long_content,
                match_result=sample_match_result,
                changes=sample_changes
            )


class TestPDFGenerator:
    """Test cases for PDFGenerator class."""

    def test_pdf_generator_initialization_default(self):
        """Test default initialization of PDFGenerator."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()
            assert generator.options.format == "A4"
            assert generator.options.margin_top == "1cm"
            assert generator.options.timeout == 30000

    def test_pdf_generator_initialization_custom(self):
        """Test custom initialization of PDFGenerator."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            options = PDFOptions(
                format="Letter",
                margin_top="2cm",
                timeout=45000,
                scale=0.9
            )
            generator = PDFGenerator(options)
            assert generator.options.format == "Letter"
            assert generator.options.margin_top == "2cm"
            assert generator.options.timeout == 45000
            assert generator.options.scale == 0.9

    def test_playwright_not_available_error(self):
        """Test error when Playwright is not available."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', False):
            with pytest.raises(PlaywrightNotAvailableError):
                PDFGenerator()

    def test_apply_pdf_styling(self):
        """Test PDF styling application."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            html_content = '<p>Test content</p>'
            styled_html = generator.apply_pdf_styling(html_content)

            assert '<style>' in styled_html
            assert '@page' in styled_html
            assert 'font-family' in styled_html
            assert html_content in styled_html

    def test_apply_pdf_styling_with_existing_html(self):
        """Test PDF styling with existing HTML structure."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            html_with_head = '<html><head><title>Test</title></head><body><p>Content</p></body></html>'
            styled_html = generator.apply_pdf_styling(html_with_head)

            assert '<style>' in styled_html
            assert 'Content' in styled_html

    def test_validate_pdf_output_nonexistent_file(self):
        """Test PDF validation with non-existent file."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            result = generator.validate_pdf_output('/nonexistent/path/test.pdf')
            assert result is False

    def test_validate_pdf_output_valid_pdf(self):
        """Test PDF validation with valid PDF file."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            # Create a temporary file with PDF header
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(b'%PDF-1.4\ntest content for size')
                tmp_file.flush()

                result = generator.validate_pdf_output(tmp_file.name)
                assert result is True

                # Cleanup
                Path(tmp_file.name).unlink()

    def test_generate_pdf_filename(self):
        """Test PDF filename generation."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            filename = generator._generate_pdf_filename('cv', 'Test Company', '15-12-2024')
            assert filename == 'cv_test_company_15-12-2024.pdf'

            # Test with special characters
            filename = generator._generate_pdf_filename('cover_letter', 'Test & Co.', '15-12-2024')
            assert 'test_co' in filename
            assert filename.endswith('.pdf')

    def test_validation_errors(self, sample_job_data):
        """Test input validation errors."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            # Test invalid approved_html type
            with pytest.raises(ValidationError):
                generator._validate_inputs("invalid", sample_job_data)

            # Test invalid job_data type
            with pytest.raises(ValidationError):
                generator._validate_inputs({'cv': 'content', 'cover_letter': 'content'}, "invalid")

            # Test missing required keys
            with pytest.raises(ValidationError):
                generator._validate_inputs({'invalid_key': 'content'}, sample_job_data)

            # Test empty content
            with pytest.raises(ValidationError):
                generator._validate_inputs({'cv': '', 'cover_letter': ''}, sample_job_data)

    def test_css_generation(self):
        """Test PDF CSS generation."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            generator = PDFGenerator()

            css = generator._generate_pdf_css()

            assert '@page' in css
            assert 'font-family' in css
            assert 'A4' in css
            assert 'margin' in css

    @pytest.mark.asyncio
    async def test_browser_context_manager(self):
        """Test browser context manager functionality."""
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            with patch('src.agents.pdf_generator.async_playwright') as mock_playwright:
                # Mock the playwright context
                mock_browser = Mock()
                mock_playwright_instance = Mock()
                mock_playwright_instance.chromium.launch.return_value = mock_browser
                mock_playwright.return_value.start.return_value = mock_playwright_instance

                generator = PDFGenerator()

                async with generator:
                    assert generator._browser is not None

                # Verify cleanup was called
                mock_browser.close.assert_called_once()


class TestIntegration:
    """Integration tests for the complete system."""

    def test_complete_workflow_preview_only(self, sample_html_content, sample_match_result,
                                          sample_changes, sample_original_content):
        """Test complete workflow for preview generation."""
        # Initialize generators
        preview_generator = PreviewGenerator(PreviewOptions(
            show_side_by_side=True,
            include_match_analysis=True,
            enable_interactive_approval=True
        ))

        # Generate preview
        preview_html = preview_generator.generate_preview(
            customized_html=sample_html_content,
            match_result=sample_match_result,
            changes=sample_changes,
            original_html=sample_original_content
        )

        # Verify complete HTML structure
        assert '<!DOCTYPE html>' in preview_html
        assert '<html lang="en">' in preview_html
        assert '</html>' in preview_html

        # Verify all major sections are present
        expected_sections = [
            'preview-header',
            'match-analysis',
            'changes-summary',
            'comparisons-section',
            'documents-section',
            'approval-interface',
            'preview-footer'
        ]

        for section in expected_sections:
            assert section in preview_html

    def test_error_handling_chain(self, sample_job_data):
        """Test error handling across the system."""
        # Test PreviewGenerator errors
        preview_generator = PreviewGenerator()

        with pytest.raises(ValidationError):
            preview_generator.generate_preview(
                customized_html={'invalid': 'content'},
                match_result=sample_job_data,  # Wrong type
                changes=[]
            )

        # Test PDFGenerator errors
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', True):
            pdf_generator = PDFGenerator()

            with pytest.raises(ValidationError):
                pdf_generator._validate_inputs({'invalid': 'content'}, sample_job_data)


class TestPerformance:
    """Performance-related tests."""

    def test_large_content_handling(self, sample_match_result):
        """Test handling of large content."""
        # Create large content
        large_content = {
            'cv': '<p>' + 'x' * 5000 + '</p>',
            'cover_letter': '<p>' + 'y' * 5000 + '</p>'
        }

        changes = ['Large content test']

        generator = PreviewGenerator()

        # Should handle large content without issues
        preview_html = generator.generate_preview(
            customized_html=large_content,
            match_result=sample_match_result,
            changes=changes
        )

        assert len(preview_html) > 10000  # Should generate substantial HTML

    def test_memory_usage_preview(self, sample_html_content, sample_match_result, sample_changes):
        """Test memory usage during preview generation."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        generator = PreviewGenerator()

        # Generate multiple previews
        for _ in range(10):
            preview_html = generator.generate_preview(
                customized_html=sample_html_content,
                match_result=sample_match_result,
                changes=sample_changes
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 10 previews)
        assert memory_increase < 50 * 1024 * 1024


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])