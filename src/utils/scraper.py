"""
Web scraping utility for job postings with comprehensive error handling.

This module provides robust web scraping capabilities specifically designed for
extracting job posting content from various websites with proper error handling,
user-agent rotation, and comprehensive logging.
"""

import asyncio
import logging
import random
import re
import time
from typing import Optional, Dict, List
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import ssl

from .logging_config import get_scraping_logger

logger = get_scraping_logger()

# Common user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
]

# Request configuration
REQUEST_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 2,
    "max_content_length": 10 * 1024 * 1024,  # 10MB
}


class ScrapingError(Exception):
    """Custom exception for scraping-related errors."""
    pass


class ScrapingTimeoutError(ScrapingError):
    """Exception raised when scraping times out."""
    pass


class ScrapingBlockedError(ScrapingError):
    """Exception raised when scraping is blocked by the website."""
    pass


def validate_url(url: str) -> bool:
    """
    Validate if URL is properly formatted and accessible.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    try:
        if not url or not isinstance(url, str):
            return False

        # Basic URL format validation
        parsed = urlparse(url.strip())

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False

        # Must be HTTP or HTTPS
        if parsed.scheme.lower() not in ['http', 'https']:
            return False

        # Basic domain validation
        domain = parsed.netloc.lower()
        if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain):
            return False

        logger.debug(f"URL validation successful: {url}")
        return True

    except Exception as e:
        logger.warning(f"URL validation failed for '{url}': {e}")
        return False


def _get_random_headers() -> Dict[str, str]:
    """
    Generate randomized headers with user-agent rotation.

    Returns:
        Dictionary of HTTP headers
    """
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def _detect_blocked_content(response: requests.Response, content: str) -> bool:
    """
    Detect if the response indicates blocked or restricted access.

    Args:
        response: HTTP response object
        content: Response content as string

    Returns:
        True if content appears to be blocked
    """
    # Check status codes that indicate blocking
    blocked_status_codes = [403, 429, 503]
    if response.status_code in blocked_status_codes:
        return True

    # Check for common blocking indicators in content
    blocking_indicators = [
        "access denied",
        "blocked",
        "captcha",
        "rate limit",
        "too many requests",
        "cloudflare",
        "please enable javascript",
        "bot detection",
        "human verification"
    ]

    content_lower = content.lower()
    return any(indicator in content_lower for indicator in blocking_indicators)


def extract_text_content(html: str) -> str:
    """
    Extract clean text content from HTML.

    Args:
        html: Raw HTML content

    Returns:
        Cleaned text content

    Raises:
        ScrapingError: If HTML parsing fails
    """
    try:
        if not html or not isinstance(html, str):
            raise ScrapingError("Invalid HTML content provided")

        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        logger.debug(f"Extracted {len(text)} characters of text content")
        return text

    except Exception as e:
        error_msg = f"Failed to extract text content: {e}"
        logger.error(error_msg)
        raise ScrapingError(error_msg)


def scrape_job_posting(url: str) -> str:
    """
    Scrape job posting content from a URL with comprehensive error handling.

    Args:
        url: URL of the job posting to scrape

    Returns:
        Extracted text content from the job posting

    Raises:
        ScrapingError: For general scraping errors
        ScrapingTimeoutError: For timeout errors
        ScrapingBlockedError: When access is blocked
    """
    logger.info(f"Starting scraping attempt for URL: {url}")

    # Validate URL first
    if not validate_url(url):
        raise ScrapingError(f"Invalid URL provided: {url}")

    session = requests.Session()
    last_exception = None

    for attempt in range(REQUEST_CONFIG["max_retries"]):
        try:
            logger.debug(f"Scraping attempt {attempt + 1}/{REQUEST_CONFIG['max_retries']} for {url}")

            # Random delay between attempts
            if attempt > 0:
                delay = REQUEST_CONFIG["retry_delay"] * (attempt + random.uniform(0.5, 1.5))
                logger.debug(f"Waiting {delay:.1f} seconds before retry")
                time.sleep(delay)

            # Configure request
            headers = _get_random_headers()

            # Make request
            response = session.get(
                url,
                headers=headers,
                timeout=REQUEST_CONFIG["timeout"],
                allow_redirects=True,
                verify=True  # SSL verification
            )

            # Check content length
            content_length = len(response.content)
            if content_length > REQUEST_CONFIG["max_content_length"]:
                raise ScrapingError(f"Content too large: {content_length} bytes")

            # Check for successful response
            if response.status_code == 200:
                content = response.text

                # Check if content is blocked
                if _detect_blocked_content(response, content):
                    logger.warning(f"Blocked content detected for {url}")
                    raise ScrapingBlockedError("Access appears to be blocked by the website")

                # Extract text content
                text_content = extract_text_content(content)

                if len(text_content.strip()) < 50:
                    raise ScrapingError("Extracted content is too short, possibly empty page")

                logger.info(f"Successfully scraped {len(text_content)} characters from {url}")
                return text_content

            elif response.status_code == 404:
                raise ScrapingError(f"Job posting not found (404): {url}")

            elif response.status_code in [403, 429, 503]:
                raise ScrapingBlockedError(f"Access blocked (HTTP {response.status_code}): {url}")

            else:
                raise ScrapingError(f"HTTP {response.status_code}: {response.reason}")

        except requests.exceptions.Timeout:
            last_exception = ScrapingTimeoutError(f"Request timeout after {REQUEST_CONFIG['timeout']} seconds")
            logger.warning(f"Timeout on attempt {attempt + 1}: {url}")

        except requests.exceptions.ConnectionError as e:
            last_exception = ScrapingError(f"Connection error: {e}")
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")

        except requests.exceptions.SSLError as e:
            last_exception = ScrapingError(f"SSL error: {e}")
            logger.warning(f"SSL error on attempt {attempt + 1}: {e}")

        except (ScrapingBlockedError, ScrapingError):
            # Re-raise immediately, don't retry
            raise

        except Exception as e:
            last_exception = ScrapingError(f"Unexpected error: {e}")
            logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")

    # All retries exhausted
    error_msg = f"Failed to scrape {url} after {REQUEST_CONFIG['max_retries']} attempts"
    logger.error(error_msg)

    if last_exception:
        raise last_exception
    else:
        raise ScrapingError(error_msg)


def get_domain_info(url: str) -> Dict[str, str]:
    """
    Extract domain information from URL.

    Args:
        url: URL to analyze

    Returns:
        Dictionary with domain information
    """
    try:
        parsed = urlparse(url)
        return {
            "domain": parsed.netloc,
            "scheme": parsed.scheme,
            "path": parsed.path,
            "full_url": url
        }
    except Exception as e:
        logger.warning(f"Failed to parse domain info from {url}: {e}")
        return {"domain": "unknown", "scheme": "", "path": "", "full_url": url}


def is_supported_domain(url: str) -> bool:
    """
    Check if the domain is known to work well with scraping.

    Args:
        url: URL to check

    Returns:
        True if domain is supported, False otherwise
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Known working domains
        supported_patterns = [
            r'.*\.jobs$',
            r'.*\.careers$',
            r'lever\.co$',
            r'greenhouse\.io$',
            r'workday\.com$',
            r'indeed\.com$',
            r'linkedin\.com$',
        ]

        return any(re.match(pattern, domain) for pattern in supported_patterns)

    except Exception:
        return False


def batch_scrape_urls(urls: List[str], max_concurrent: int = 3) -> Dict[str, str]:
    """
    Scrape multiple URLs concurrently with rate limiting.

    Args:
        urls: List of URLs to scrape
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Dictionary mapping URLs to their scraped content
    """
    logger.info(f"Starting batch scraping of {len(urls)} URLs")

    results = {}

    for i, url in enumerate(urls):
        try:
            logger.debug(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = scrape_job_posting(url)
            results[url] = content

            # Rate limiting
            if i < len(urls) - 1:  # Don't delay after the last URL
                time.sleep(random.uniform(1, 3))

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            results[url] = f"ERROR: {str(e)}"

    logger.info(f"Batch scraping completed. Success: {sum(1 for v in results.values() if not v.startswith('ERROR:'))}/{len(urls)}")
    return results