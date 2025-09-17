"""
JobAnalyzer class for comprehensive job posting analysis using AI.

This module provides intelligent job posting analysis capabilities, combining
web scraping with OpenAI GPT integration to extract structured data from job
postings. It handles both URLs and raw text input with comprehensive error
handling and validation.
"""

import json
import os
import time
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import ValidationError as PydanticValidationError

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.job_data import JobData, ExperienceLevel
from utils.scraper import scrape_job_posting, validate_url, ScrapingError, ScrapingTimeoutError, ScrapingBlockedError
from utils.logging_config import get_agents_logger

logger = get_agents_logger()


class JobAnalysisError(Exception):
    """Base exception for job analysis errors."""
    pass


class OpenAIConfigurationError(JobAnalysisError):
    """Exception raised when OpenAI configuration is invalid."""
    pass


class ExtractionError(JobAnalysisError):
    """Exception raised when data extraction fails."""
    pass


class JobDataValidationError(JobAnalysisError):
    """Exception raised when extracted data validation fails."""
    pass


class JobAnalyzer:
    """
    Comprehensive job posting analyzer using AI and web scraping.

    This class provides intelligent analysis of job postings from both URLs
    and raw text input. It uses OpenAI GPT models to extract structured data
    and validates the results for consistency and completeness.

    Attributes:
        openai_client: OpenAI client instance for API calls
        model: GPT model to use for analysis (default: gpt-4)
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay between retries in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the JobAnalyzer with OpenAI configuration.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: GPT model to use for analysis
            max_retries: Maximum retry attempts for API failures
            retry_delay: Base delay between retries in seconds

        Raises:
            OpenAIConfigurationError: If API key is not provided or invalid
        """
        logger.info("Initializing JobAnalyzer")

        # Configure OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise OpenAIConfigurationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            self.openai_client = OpenAI(api_key=api_key)
            self.model = model
            self.max_retries = max_retries
            self.retry_delay = retry_delay

            logger.info(f"JobAnalyzer initialized with model: {model}, max_retries: {max_retries}")

        except Exception as e:
            raise OpenAIConfigurationError(f"Failed to initialize OpenAI client: {e}")

    def _is_url(self, input_text: str) -> bool:
        """
        Detect if input is a URL or raw text.

        Args:
            input_text: Input string to analyze

        Returns:
            True if input appears to be a URL, False otherwise
        """
        if not input_text or not isinstance(input_text, str):
            return False

        # Clean the input
        cleaned_input = input_text.strip()

        # Check if it starts with http/https
        if cleaned_input.lower().startswith(('http://', 'https://')):
            return validate_url(cleaned_input)

        # Check if it looks like a URL without protocol
        url_pattern = r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
        if re.match(url_pattern, cleaned_input):
            # Try with https prefix
            test_url = f"https://{cleaned_input}"
            return validate_url(test_url)

        return False

    def _create_extraction_prompt(self, job_text: str) -> str:
        """
        Create an optimized prompt for job data extraction.

        Args:
            job_text: Raw job posting text

        Returns:
            Formatted prompt for GPT
        """
        return f"""
You are a professional job posting analyzer. Extract structured information from the following job posting and return it as valid JSON.

REQUIRED OUTPUT FORMAT:
{{
    "company_name": "string - Company name (if not found, use 'Unknown Company')",
    "position": "string - Job title/position",
    "requirements": ["string array - List of job requirements and qualifications"],
    "skills_required": ["string array - Technical and soft skills needed"],
    "experience_level": "string - One of: entry, junior, mid, senior, lead, executive, intern",
    "description": "string - Clean, comprehensive job description"
}}

EXTRACTION GUIDELINES:
1. company_name: Extract the hiring company name. If unclear, use "Unknown Company"
2. position: Extract the main job title. Be concise but descriptive
3. requirements: Extract specific qualifications, education, certifications, experience needed
4. skills_required: Extract technical skills, programming languages, tools, soft skills
5. experience_level: Determine based on years of experience, seniority level, job title
6. description: Create a clean, comprehensive description combining key information

EXPERIENCE LEVEL MAPPING:
- "entry": 0-1 years, entry-level, graduate, new grad
- "junior": 1-3 years, junior, associate
- "mid": 3-5 years, mid-level, experienced
- "senior": 5-8 years, senior, lead developer/engineer
- "lead": 8+ years, team lead, principal, architect
- "executive": C-level, director, VP, executive roles
- "intern": Internship, co-op, student positions

IMPORTANT:
- Return ONLY valid JSON, no additional text
- Ensure all strings are properly escaped
- If information is missing, use reasonable defaults
- Keep arrays non-empty when possible
- Make description at least 50 characters

JOB POSTING:
{job_text}
"""

    def _call_openai_with_retry(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API with exponential backoff retry logic.

        Args:
            prompt: The prompt to send to OpenAI

        Returns:
            Parsed JSON response from OpenAI

        Raises:
            ExtractionError: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"OpenAI API call attempt {attempt + 1}/{self.max_retries}")

                # Add delay for retries
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.debug(f"Waiting {delay} seconds before retry")
                    time.sleep(delay)

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional job posting analyzer. Extract structured information and return valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=2000,
                    timeout=30
                )

                # Extract content
                content = response.choices[0].message.content
                if not content:
                    raise ExtractionError("Empty response from OpenAI")

                # Parse JSON
                try:
                    extracted_data = json.loads(content.strip())
                    logger.debug("Successfully parsed JSON response from OpenAI")
                    return extracted_data

                except json.JSONDecodeError as e:
                    # Try to extract JSON from response if it's wrapped in text
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            extracted_data = json.loads(json_match.group())
                            logger.debug("Successfully extracted and parsed JSON from response")
                            return extracted_data
                        except json.JSONDecodeError:
                            pass

                    raise ExtractionError(f"Failed to parse JSON response: {e}")

            except Exception as e:
                last_exception = e
                logger.warning(f"OpenAI API call failed on attempt {attempt + 1}: {e}")

                # Don't retry for certain types of errors
                if "invalid_api_key" in str(e).lower() or "quota" in str(e).lower():
                    raise ExtractionError(f"OpenAI API error: {e}")

        # All retries exhausted
        error_msg = f"Failed to call OpenAI API after {self.max_retries} attempts"
        logger.error(error_msg)

        if last_exception:
            raise ExtractionError(f"{error_msg}: {last_exception}")
        else:
            raise ExtractionError(error_msg)

    def extract_job_details(self, job_text: str) -> JobData:
        """
        Extract structured job data from raw text using OpenAI GPT.

        Args:
            job_text: Raw job posting text

        Returns:
            JobData instance with extracted information

        Raises:
            ExtractionError: If extraction or validation fails
        """
        logger.info("Starting job details extraction from text")

        if not job_text or not isinstance(job_text, str):
            raise ExtractionError("Invalid job text provided")

        # Clean the input text
        cleaned_text = job_text.strip()
        if len(cleaned_text) < 20:
            raise ExtractionError("Job text too short for meaningful analysis")

        # Create extraction prompt
        prompt = self._create_extraction_prompt(cleaned_text)

        # Call OpenAI API
        extracted_data = self._call_openai_with_retry(prompt)

        # Validate required fields
        required_fields = ['company_name', 'position', 'description']
        for field in required_fields:
            if field not in extracted_data or not extracted_data[field]:
                logger.warning(f"Missing required field: {field}")
                if field == 'company_name':
                    extracted_data[field] = 'Unknown Company'
                elif field == 'position':
                    extracted_data[field] = 'Unknown Position'
                elif field == 'description':
                    extracted_data[field] = cleaned_text[:500]  # Use original text as fallback

        # Validate experience level
        if 'experience_level' in extracted_data:
            exp_level = extracted_data['experience_level'].lower()
            valid_levels = [level.value for level in ExperienceLevel]
            if exp_level not in valid_levels:
                logger.warning(f"Invalid experience level: {exp_level}, defaulting to 'mid'")
                extracted_data['experience_level'] = 'mid'
        else:
            extracted_data['experience_level'] = 'mid'

        # Ensure lists are present
        for list_field in ['requirements', 'skills_required']:
            if list_field not in extracted_data or not isinstance(extracted_data[list_field], list):
                extracted_data[list_field] = []

        try:
            # Create JobData instance
            job_data = JobData(**extracted_data)
            logger.info(f"Successfully extracted job data: {job_data.get_summary()}")
            return job_data

        except PydanticValidationError as e:
            error_msg = f"Failed to validate extracted job data: {e}"
            logger.error(error_msg)
            raise ExtractionError(error_msg)

    def validate_extraction(self, job_data: JobData) -> bool:
        """
        Validate the quality and completeness of extracted job data.

        Args:
            job_data: JobData instance to validate

        Returns:
            True if data meets quality standards, False otherwise
        """
        logger.debug("Validating extracted job data quality")

        issues = []

        # Check company name quality
        if job_data.company_name.lower() in ['unknown company', 'company', 'n/a', 'tbd']:
            issues.append("Company name appears to be a placeholder")

        # Check position quality
        if job_data.position.lower() in ['unknown position', 'position', 'job', 'role']:
            issues.append("Position appears to be a placeholder")

        # Check description length and quality
        if len(job_data.description) < 50:
            issues.append("Description is too short")

        # Check for reasonable number of skills and requirements
        if len(job_data.skills_required) == 0:
            issues.append("No skills extracted")
        elif len(job_data.skills_required) > 50:
            issues.append("Unusually high number of skills (possible extraction error)")

        if len(job_data.requirements) == 0:
            issues.append("No requirements extracted")
        elif len(job_data.requirements) > 30:
            issues.append("Unusually high number of requirements (possible extraction error)")

        # Check for duplicates in skills and requirements
        unique_skills = set(skill.lower().strip() for skill in job_data.skills_required)
        if len(unique_skills) < len(job_data.skills_required) * 0.8:  # Allow for some variation
            issues.append("High number of duplicate skills detected")

        # Log issues if any
        if issues:
            logger.warning(f"Validation issues found: {', '.join(issues)}")
            return False

        logger.debug("Job data validation passed")
        return True

    def analyze_job(self, job_input: str) -> JobData:
        """
        Main entry point for job analysis. Handles both URLs and raw text.

        Args:
            job_input: Either a URL to a job posting or raw job text

        Returns:
            JobData instance with extracted and validated information

        Raises:
            JobAnalysisError: If analysis fails for any reason
        """
        logger.info("Starting job analysis")

        if not job_input or not isinstance(job_input, str):
            raise JobAnalysisError("Invalid job input provided")

        job_text = job_input.strip()

        # Determine if input is URL or text
        if self._is_url(job_input):
            logger.info(f"Input detected as URL: {job_input}")

            try:
                # Scrape job posting content
                job_text = scrape_job_posting(job_input)
                logger.info(f"Successfully scraped {len(job_text)} characters from URL")

            except ScrapingBlockedError as e:
                error_msg = f"Job posting access blocked: {e}"
                logger.error(error_msg)
                raise JobAnalysisError(f"{error_msg}. Please try copying the job text directly.")

            except ScrapingTimeoutError as e:
                error_msg = f"Job posting scraping timed out: {e}"
                logger.error(error_msg)
                raise JobAnalysisError(f"{error_msg}. Please try again or copy the job text directly.")

            except ScrapingError as e:
                error_msg = f"Failed to scrape job posting: {e}"
                logger.error(error_msg)
                raise JobAnalysisError(f"{error_msg}. Please try copying the job text directly.")

        else:
            logger.info("Input detected as raw text")

        # Extract job details using AI
        try:
            job_data = self.extract_job_details(job_text)

            # Add URL if it was provided
            if self._is_url(job_input):
                job_data.url = job_input

            # Validate extraction quality
            is_valid = self.validate_extraction(job_data)
            if not is_valid:
                logger.warning("Job data validation failed, but continuing with extracted data")

            logger.info(f"Job analysis completed successfully: {job_data.get_summary()}")
            return job_data

        except ExtractionError as e:
            error_msg = f"Job analysis failed: {e}"
            logger.error(error_msg)
            raise JobAnalysisError(error_msg)

    def get_analysis_summary(self, job_data: JobData) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis summary.

        Args:
            job_data: JobData instance to summarize

        Returns:
            Dictionary containing analysis summary and metadata
        """
        logger.debug("Generating analysis summary")

        return {
            "job_summary": job_data.get_summary(),
            "extraction_timestamp": job_data.extracted_at.isoformat(),
            "company": job_data.company_name,
            "position": job_data.position,
            "experience_level": job_data.experience_level,
            "skills_count": len(job_data.skills_required),
            "requirements_count": len(job_data.requirements),
            "top_skills": job_data.skills_required[:10],  # Top 10 skills
            "source_url": str(job_data.url) if job_data.url else None,
            "description_length": len(job_data.description),
            "validation_passed": self.validate_extraction(job_data)
        }