"""
Comprehensive tests for the JobAnalyzer class.

This module contains unit tests for all functionality of the JobAnalyzer class,
including successful analysis, error handling, validation, and edge cases.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.agents.job_analyzer import (
    JobAnalyzer,
    JobAnalysisError,
    OpenAIConfigurationError,
    ExtractionError,
    JobDataValidationError
)
from src.models.job_data import JobData, ExperienceLevel
from src.utils.scraper import ScrapingError, ScrapingTimeoutError, ScrapingBlockedError


class TestJobAnalyzerInitialization:
    """Test JobAnalyzer initialization and configuration."""

    def test_initialization_with_api_key(self):
        """Test successful initialization with API key."""
        with patch('src.agents.job_analyzer.OpenAI') as mock_openai:
            analyzer = JobAnalyzer(api_key="test-key")

            assert analyzer.model == "gpt-4"
            assert analyzer.max_retries == 3
            assert analyzer.retry_delay == 2
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_initialization_with_env_var(self):
        """Test initialization using environment variable."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            with patch('src.agents.job_analyzer.OpenAI') as mock_openai:
                analyzer = JobAnalyzer()
                mock_openai.assert_called_once_with(api_key="env-key")

    def test_initialization_without_api_key(self):
        """Test initialization failure without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(OpenAIConfigurationError) as exc_info:
                JobAnalyzer()

            assert "OpenAI API key not provided" in str(exc_info.value)

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        with patch('src.agents.job_analyzer.OpenAI'):
            analyzer = JobAnalyzer(
                api_key="test-key",
                model="gpt-3.5-turbo",
                max_retries=5,
                retry_delay=3
            )

            assert analyzer.model == "gpt-3.5-turbo"
            assert analyzer.max_retries == 5
            assert analyzer.retry_delay == 3


class TestURLDetection:
    """Test URL detection functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_is_url_with_https(self, analyzer):
        """Test URL detection with HTTPS."""
        with patch('src.agents.job_analyzer.validate_url', return_value=True):
            assert analyzer._is_url("https://example.com/job/123") is True

    def test_is_url_with_http(self, analyzer):
        """Test URL detection with HTTP."""
        with patch('src.agents.job_analyzer.validate_url', return_value=True):
            assert analyzer._is_url("http://example.com/job/123") is True

    def test_is_url_without_protocol(self, analyzer):
        """Test URL detection without protocol."""
        with patch('src.agents.job_analyzer.validate_url', return_value=True):
            assert analyzer._is_url("example.com/job/123") is True

    def test_is_not_url_plain_text(self, analyzer):
        """Test that plain text is not detected as URL."""
        assert analyzer._is_url("This is a job description") is False

    def test_is_not_url_empty_string(self, analyzer):
        """Test that empty string is not detected as URL."""
        assert analyzer._is_url("") is False

    def test_is_not_url_none(self, analyzer):
        """Test that None is not detected as URL."""
        assert analyzer._is_url(None) is False


class TestExtractionPrompt:
    """Test extraction prompt generation."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_create_extraction_prompt(self, analyzer):
        """Test extraction prompt creation."""
        job_text = "Software Engineer position at TechCorp"
        prompt = analyzer._create_extraction_prompt(job_text)

        assert "Software Engineer position at TechCorp" in prompt
        assert "company_name" in prompt
        assert "position" in prompt
        assert "requirements" in prompt
        assert "skills_required" in prompt
        assert "experience_level" in prompt
        assert "description" in prompt
        assert "entry, junior, mid, senior, lead, executive, intern" in prompt


class TestOpenAIIntegration:
    """Test OpenAI API integration."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI') as mock_openai:
            analyzer = JobAnalyzer(api_key="test-key")
            analyzer.openai_client = mock_openai.return_value
            return analyzer

    def test_successful_openai_call(self, analyzer):
        """Test successful OpenAI API call."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "company_name": "TechCorp",
            "position": "Software Engineer",
            "requirements": ["Bachelor's degree", "3 years experience"],
            "skills_required": ["Python", "JavaScript"],
            "experience_level": "mid",
            "description": "Great opportunity for a software engineer"
        })

        analyzer.openai_client.chat.completions.create.return_value = mock_response

        result = analyzer._call_openai_with_retry("test prompt")

        assert result["company_name"] == "TechCorp"
        assert result["position"] == "Software Engineer"
        assert "Python" in result["skills_required"]

    def test_openai_call_with_retry(self, analyzer):
        """Test OpenAI API call with retry logic."""
        # Mock first call to fail, second to succeed
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "company_name": "TechCorp",
            "position": "Software Engineer",
            "requirements": [],
            "skills_required": [],
            "experience_level": "mid",
            "description": "Test description"
        })

        analyzer.openai_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            mock_response
        ]

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = analyzer._call_openai_with_retry("test prompt")

        assert result["company_name"] == "TechCorp"
        assert analyzer.openai_client.chat.completions.create.call_count == 2

    def test_openai_call_max_retries_exceeded(self, analyzer):
        """Test OpenAI API call when max retries exceeded."""
        analyzer.openai_client.chat.completions.create.side_effect = Exception("API Error")

        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(ExtractionError) as exc_info:
                analyzer._call_openai_with_retry("test prompt")

        assert "Failed to call OpenAI API after 3 attempts" in str(exc_info.value)

    def test_openai_call_json_parsing_error(self, analyzer):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"

        analyzer.openai_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ExtractionError) as exc_info:
            analyzer._call_openai_with_retry("test prompt")

        assert "Failed to parse JSON response" in str(exc_info.value)

    def test_openai_call_json_extraction_from_text(self, analyzer):
        """Test JSON extraction from text response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        Here is the extracted data:
        {
            "company_name": "TechCorp",
            "position": "Software Engineer",
            "requirements": [],
            "skills_required": [],
            "experience_level": "mid",
            "description": "Test description"
        }
        That's the result.
        """

        analyzer.openai_client.chat.completions.create.return_value = mock_response

        result = analyzer._call_openai_with_retry("test prompt")
        assert result["company_name"] == "TechCorp"


class TestJobDataExtraction:
    """Test job data extraction functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_extract_job_details_success(self, analyzer):
        """Test successful job details extraction."""
        job_text = "Software Engineer position at TechCorp. Requirements: Python, 3 years experience."

        # Mock OpenAI response
        mock_data = {
            "company_name": "TechCorp",
            "position": "Software Engineer",
            "requirements": ["3 years experience"],
            "skills_required": ["Python"],
            "experience_level": "mid",
            "description": job_text
        }

        with patch.object(analyzer, '_call_openai_with_retry', return_value=mock_data):
            result = analyzer.extract_job_details(job_text)

        assert isinstance(result, JobData)
        assert result.company_name == "TechCorp"
        assert result.position == "Software Engineer"
        assert "Python" in result.skills_required

    def test_extract_job_details_empty_input(self, analyzer):
        """Test extraction with empty input."""
        with pytest.raises(ExtractionError) as exc_info:
            analyzer.extract_job_details("")

        assert "Invalid job text provided" in str(exc_info.value)

    def test_extract_job_details_short_input(self, analyzer):
        """Test extraction with very short input."""
        with pytest.raises(ExtractionError) as exc_info:
            analyzer.extract_job_details("Short")

        assert "Job text too short" in str(exc_info.value)

    def test_extract_job_details_missing_fields(self, analyzer):
        """Test extraction with missing required fields."""
        job_text = "Some job description"

        # Mock response missing required fields
        mock_data = {
            "requirements": [],
            "skills_required": ["Python"],
            "experience_level": "mid"
        }

        with patch.object(analyzer, '_call_openai_with_retry', return_value=mock_data):
            result = analyzer.extract_job_details(job_text)

        # Should use defaults for missing fields
        assert result.company_name == "Unknown Company"
        assert result.position == "Unknown Position"
        assert len(result.description) > 0

    def test_extract_job_details_invalid_experience_level(self, analyzer):
        """Test extraction with invalid experience level."""
        job_text = "This is a longer job description that meets the minimum length requirement."

        mock_data = {
            "company_name": "TechCorp",
            "position": "Engineer",
            "requirements": [],
            "skills_required": [],
            "experience_level": "invalid_level",
            "description": job_text
        }

        with patch.object(analyzer, '_call_openai_with_retry', return_value=mock_data):
            result = analyzer.extract_job_details(job_text)

        # Should default to 'mid' for invalid experience level
        assert result.experience_level == ExperienceLevel.MID


class TestValidation:
    """Test job data validation functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_validate_extraction_success(self, analyzer):
        """Test successful validation."""
        job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=["Bachelor's degree", "3 years experience"],
            skills_required=["Python", "JavaScript", "React"],
            experience_level=ExperienceLevel.MID,
            description="A great opportunity for a software engineer to join our team."
        )

        assert analyzer.validate_extraction(job_data) is True

    def test_validate_extraction_placeholder_company(self, analyzer):
        """Test validation with placeholder company name."""
        job_data = JobData(
            company_name="Unknown Company",
            position="Software Engineer",
            requirements=["Bachelor's degree"],
            skills_required=["Python"],
            experience_level=ExperienceLevel.MID,
            description="A great opportunity for a software engineer to join our team."
        )

        assert analyzer.validate_extraction(job_data) is False

    def test_validate_extraction_short_description(self, analyzer):
        """Test validation with short description."""
        job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=["Bachelor's degree"],
            skills_required=["Python"],
            experience_level=ExperienceLevel.MID,
            description="Short desc"
        )

        assert analyzer.validate_extraction(job_data) is False

    def test_validate_extraction_no_skills(self, analyzer):
        """Test validation with no skills."""
        job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=["Bachelor's degree"],
            skills_required=[],
            experience_level=ExperienceLevel.MID,
            description="A great opportunity for a software engineer to join our team."
        )

        assert analyzer.validate_extraction(job_data) is False

    def test_validate_extraction_excessive_skills(self, analyzer):
        """Test validation with excessive number of skills."""
        job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=["Bachelor's degree"],
            skills_required=[f"Skill{i}" for i in range(60)],  # Too many skills
            experience_level=ExperienceLevel.MID,
            description="A great opportunity for a software engineer to join our team."
        )

        assert analyzer.validate_extraction(job_data) is False


class TestAnalyzeJob:
    """Test main analyze_job functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_analyze_job_with_text(self, analyzer):
        """Test job analysis with text input."""
        job_text = "Software Engineer position at TechCorp. Requirements: Python, 3 years experience."

        # Mock the extraction method
        mock_job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=["3 years experience"],
            skills_required=["Python"],
            experience_level=ExperienceLevel.MID,
            description=job_text
        )

        with patch.object(analyzer, 'extract_job_details', return_value=mock_job_data):
            with patch.object(analyzer, 'validate_extraction', return_value=True):
                result = analyzer.analyze_job(job_text)

        assert isinstance(result, JobData)
        assert result.company_name == "TechCorp"

    @patch('src.agents.job_analyzer.scrape_job_posting')
    def test_analyze_job_with_url(self, mock_scrape, analyzer):
        """Test job analysis with URL input."""
        url = "https://example.com/job/123"
        scraped_text = "Software Engineer position at TechCorp"

        mock_scrape.return_value = scraped_text

        mock_job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=[],
            skills_required=["Python"],
            experience_level=ExperienceLevel.MID,
            description=scraped_text,
            url=url
        )

        with patch.object(analyzer, '_is_url', return_value=True):
            with patch.object(analyzer, 'extract_job_details', return_value=mock_job_data):
                with patch.object(analyzer, 'validate_extraction', return_value=True):
                    result = analyzer.analyze_job(url)

        assert str(result.url) == url
        mock_scrape.assert_called_once_with(url)

    @patch('src.agents.job_analyzer.scrape_job_posting')
    def test_analyze_job_scraping_blocked(self, mock_scrape, analyzer):
        """Test job analysis when scraping is blocked."""
        url = "https://example.com/job/123"
        mock_scrape.side_effect = ScrapingBlockedError("Access blocked")

        with patch.object(analyzer, '_is_url', return_value=True):
            with pytest.raises(JobAnalysisError) as exc_info:
                analyzer.analyze_job(url)

        assert "Job posting access blocked" in str(exc_info.value)

    @patch('src.agents.job_analyzer.scrape_job_posting')
    def test_analyze_job_scraping_timeout(self, mock_scrape, analyzer):
        """Test job analysis when scraping times out."""
        url = "https://example.com/job/123"
        mock_scrape.side_effect = ScrapingTimeoutError("Timeout")

        with patch.object(analyzer, '_is_url', return_value=True):
            with pytest.raises(JobAnalysisError) as exc_info:
                analyzer.analyze_job(url)

        assert "Job posting scraping timed out" in str(exc_info.value)

    def test_analyze_job_invalid_input(self, analyzer):
        """Test job analysis with invalid input."""
        with pytest.raises(JobAnalysisError) as exc_info:
            analyzer.analyze_job("")

        assert "Invalid job input provided" in str(exc_info.value)

    def test_analyze_job_extraction_failure(self, analyzer):
        """Test job analysis when extraction fails."""
        job_text = "Some job text"

        with patch.object(analyzer, 'extract_job_details', side_effect=ExtractionError("Extraction failed")):
            with pytest.raises(JobAnalysisError) as exc_info:
                analyzer.analyze_job(job_text)

        assert "Job analysis failed" in str(exc_info.value)


class TestAnalysisSummary:
    """Test analysis summary functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_get_analysis_summary(self, analyzer):
        """Test analysis summary generation."""
        job_data = JobData(
            company_name="TechCorp",
            position="Software Engineer",
            requirements=["Bachelor's degree", "3 years experience"],
            skills_required=["Python", "JavaScript", "React"],
            experience_level=ExperienceLevel.MID,
            description="A great opportunity for a software engineer to join our team.",
            url="https://example.com/job/123"
        )

        with patch.object(analyzer, 'validate_extraction', return_value=True):
            summary = analyzer.get_analysis_summary(job_data)

        assert summary["company"] == "TechCorp"
        assert summary["position"] == "Software Engineer"
        assert summary["experience_level"] == "mid"
        assert summary["skills_count"] == 3
        assert summary["requirements_count"] == 2
        assert summary["source_url"] == "https://example.com/job/123"
        assert summary["validation_passed"] is True
        assert "Python" in summary["top_skills"]


# Integration test fixtures and data
@pytest.fixture
def sample_job_text():
    """Sample job posting text for testing."""
    return """
    Software Engineer - Backend
    TechCorp Inc.

    We are looking for a talented Backend Software Engineer to join our growing team.

    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 3-5 years of experience in backend development
    - Strong proficiency in Python and Django
    - Experience with PostgreSQL and Redis
    - Knowledge of AWS services
    - Experience with Docker and Kubernetes

    Responsibilities:
    - Design and implement scalable backend services
    - Collaborate with frontend developers and product managers
    - Write clean, maintainable code with proper testing
    - Optimize application performance and scalability

    We offer competitive salary, health benefits, and flexible working arrangements.
    """


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def analyzer(self):
        """Create JobAnalyzer instance for testing."""
        with patch('src.agents.job_analyzer.OpenAI'):
            return JobAnalyzer(api_key="test-key")

    def test_end_to_end_text_analysis(self, analyzer, sample_job_text):
        """Test complete workflow with text input."""
        # Mock OpenAI response
        mock_data = {
            "company_name": "TechCorp Inc.",
            "position": "Software Engineer - Backend",
            "requirements": [
                "Bachelor's degree in Computer Science",
                "3-5 years of experience in backend development",
                "Strong proficiency in Python and Django"
            ],
            "skills_required": [
                "Python", "Django", "PostgreSQL", "Redis",
                "AWS", "Docker", "Kubernetes"
            ],
            "experience_level": "mid",
            "description": sample_job_text.strip()
        }

        with patch.object(analyzer, '_call_openai_with_retry', return_value=mock_data):
            result = analyzer.analyze_job(sample_job_text)
            summary = analyzer.get_analysis_summary(result)

        assert result.company_name == "TechCorp Inc."
        assert result.position == "Software Engineer - Backend"
        assert result.experience_level == ExperienceLevel.MID
        assert "Python" in result.skills_required
        assert "Django" in result.skills_required
        assert len(result.requirements) == 3
        assert summary["validation_passed"] is True