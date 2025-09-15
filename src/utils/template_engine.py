"""
HTML template engine utility for CV/Cover Letter generation.

This module provides template loading, placeholder replacement, and HTML processing
capabilities with comprehensive error handling and HTML format preservation.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from html import escape, unescape

from .logging_config import get_scraping_logger

logger = get_scraping_logger()


class TemplateError(Exception):
    """Custom exception for template-related errors."""
    pass


class TemplateNotFoundError(TemplateError):
    """Exception raised when template file is not found."""
    pass


class PlaceholderError(TemplateError):
    """Exception raised when placeholder processing fails."""
    pass


# Regular expressions for finding placeholders
HTML_COMMENT_PLACEHOLDER_PATTERN = re.compile(r'<!--\s*([A-Z_][A-Z0-9_]*)\s*-->', re.IGNORECASE)
DOUBLE_BRACE_PLACEHOLDER_PATTERN = re.compile(r'\{\{\s*([A-Z_][A-Z0-9_]*)\s*\}\}', re.IGNORECASE)
SINGLE_BRACE_PLACEHOLDER_PATTERN = re.compile(r'\{\s*([A-Z_][A-Z0-9_]*)\s*\}', re.IGNORECASE)


def load_template(template_path: str) -> str:
    """
    Load HTML template from file.

    Args:
        template_path: Path to the HTML template file

    Returns:
        Template content as string

    Raises:
        TemplateNotFoundError: If template file is not found
        TemplateError: If template loading fails
    """
    logger.info(f"Loading template from: {template_path}")

    try:
        template_file = Path(template_path)

        # Check file exists
        if not template_file.exists():
            raise TemplateNotFoundError(f"Template file not found: {template_path}")

        if not template_file.is_file():
            raise TemplateNotFoundError(f"Path is not a file: {template_path}")

        # Check file extension
        if template_file.suffix.lower() not in ['.html', '.htm']:
            logger.warning(f"Template file does not have HTML extension: {template_path}")

        # Read template content
        with open(template_file, 'r', encoding='utf-8') as file:
            content = file.read()

        if not content.strip():
            raise TemplateError(f"Template file is empty: {template_path}")

        logger.info(f"Successfully loaded template: {len(content)} characters from {template_path}")
        return content

    except FileNotFoundError:
        raise TemplateNotFoundError(f"Template file not found: {template_path}")

    except PermissionError:
        error_msg = f"Permission denied reading template: {template_path}"
        logger.error(error_msg)
        raise TemplateError(error_msg)

    except UnicodeDecodeError as e:
        error_msg = f"Unicode decode error in template {template_path}: {e}"
        logger.error(error_msg)
        raise TemplateError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error loading template: {e}"
        logger.error(error_msg)
        raise TemplateError(error_msg)


def find_placeholders(template: str, include_all_patterns: bool = True) -> List[str]:
    """
    Find all placeholders in template content.

    Args:
        template: Template content as string
        include_all_patterns: Whether to search all placeholder patterns

    Returns:
        List of unique placeholder names found in template

    Raises:
        TemplateError: If template processing fails
    """
    logger.debug("Searching for placeholders in template")

    try:
        if not isinstance(template, str):
            raise TemplateError("Template must be a string")

        placeholders = set()

        if include_all_patterns:
            # HTML comment placeholders: <!-- PLACEHOLDER_NAME -->
            html_placeholders = HTML_COMMENT_PLACEHOLDER_PATTERN.findall(template)
            placeholders.update(html_placeholders)

            # Double brace placeholders: {{ PLACEHOLDER_NAME }}
            double_brace_placeholders = DOUBLE_BRACE_PLACEHOLDER_PATTERN.findall(template)
            placeholders.update(double_brace_placeholders)

            # Single brace placeholders: { PLACEHOLDER_NAME }
            single_brace_placeholders = SINGLE_BRACE_PLACEHOLDER_PATTERN.findall(template)
            placeholders.update(single_brace_placeholders)
        else:
            # Only HTML comment placeholders (primary pattern)
            html_placeholders = HTML_COMMENT_PLACEHOLDER_PATTERN.findall(template)
            placeholders.update(html_placeholders)

        # Convert to sorted list
        placeholder_list = sorted(list(placeholders), key=str.upper)

        logger.debug(f"Found {len(placeholder_list)} unique placeholders: {placeholder_list}")
        return placeholder_list

    except Exception as e:
        error_msg = f"Error finding placeholders: {e}"
        logger.error(error_msg)
        raise TemplateError(error_msg)


def _sanitize_html_content(content: str, preserve_html: bool = True) -> str:
    """
    Sanitize content for HTML insertion.

    Args:
        content: Content to sanitize
        preserve_html: Whether to preserve existing HTML tags

    Returns:
        Sanitized content
    """
    if not preserve_html:
        # Escape all HTML
        return escape(str(content))

    # Convert to string and handle basic formatting
    content_str = str(content)

    # Convert newlines to <br> tags for HTML display
    content_str = content_str.replace('\n', '<br>\n')

    # Handle multiple spaces (preserve some formatting)
    content_str = re.sub(r'  +', lambda m: '&nbsp;' * len(m.group()), content_str)

    return content_str


def _format_list_content(content: Union[List[str], str], list_type: str = 'ul') -> str:
    """
    Format list content as HTML.

    Args:
        content: List of strings or single string
        list_type: Type of list ('ul' for unordered, 'ol' for ordered)

    Returns:
        HTML formatted list
    """
    if isinstance(content, str):
        # Split by newlines if it's a string
        items = [item.strip() for item in content.split('\n') if item.strip()]
    elif isinstance(content, list):
        items = [str(item).strip() for item in content if str(item).strip()]
    else:
        return str(content)

    if not items:
        return ''

    list_items = '\n'.join(f'    <li>{_sanitize_html_content(item)}</li>' for item in items)
    return f'<{list_type}>\n{list_items}\n</{list_type}>'


def replace_placeholders(template: str, data: Dict[str, Any], strict_mode: bool = False) -> str:
    """
    Replace placeholders in template with data values.

    Args:
        template: Template content with placeholders
        data: Dictionary of placeholder names to values
        strict_mode: If True, raise error for missing placeholders

    Returns:
        Template with placeholders replaced

    Raises:
        PlaceholderError: If placeholder replacement fails
        TemplateError: If template processing fails
    """
    logger.info(f"Replacing placeholders in template (strict_mode={strict_mode})")

    try:
        if not isinstance(template, str):
            raise TemplateError("Template must be a string")

        if not isinstance(data, dict):
            raise TemplateError("Data must be a dictionary")

        # Create case-insensitive data mapping
        data_upper = {key.upper(): value for key, value in data.items()}

        result = template
        replacement_count = 0
        missing_placeholders = []

        # Replace HTML comment placeholders first (primary pattern)
        def replace_html_comment(match):
            nonlocal replacement_count
            placeholder_name = match.group(1).upper()

            if placeholder_name in data_upper:
                replacement_count += 1
                value = data_upper[placeholder_name]

                # Handle different data types
                if isinstance(value, list):
                    return _format_list_content(value)
                elif isinstance(value, dict):
                    # Convert dict to formatted string
                    formatted_items = [f"{k}: {v}" for k, v in value.items()]
                    return _format_list_content(formatted_items)
                elif value is None:
                    return ''
                else:
                    return _sanitize_html_content(value)
            else:
                if strict_mode:
                    missing_placeholders.append(placeholder_name)
                return match.group(0)  # Leave unchanged

        result = HTML_COMMENT_PLACEHOLDER_PATTERN.sub(replace_html_comment, result)

        # Replace double brace placeholders
        def replace_double_brace(match):
            nonlocal replacement_count
            placeholder_name = match.group(1).upper()

            if placeholder_name in data_upper:
                replacement_count += 1
                value = data_upper[placeholder_name]

                if isinstance(value, list):
                    return ', '.join(str(item) for item in value)
                elif isinstance(value, dict):
                    return ', '.join(f"{k}: {v}" for k, v in value.items())
                elif value is None:
                    return ''
                else:
                    return _sanitize_html_content(value, preserve_html=False)
            else:
                if strict_mode:
                    missing_placeholders.append(placeholder_name)
                return match.group(0)

        result = DOUBLE_BRACE_PLACEHOLDER_PATTERN.sub(replace_double_brace, result)

        # Replace single brace placeholders
        def replace_single_brace(match):
            nonlocal replacement_count
            placeholder_name = match.group(1).upper()

            if placeholder_name in data_upper:
                replacement_count += 1
                value = data_upper[placeholder_name]

                if isinstance(value, (list, dict)):
                    return str(value)
                elif value is None:
                    return ''
                else:
                    return str(value)
            else:
                if strict_mode:
                    missing_placeholders.append(placeholder_name)
                return match.group(0)

        result = SINGLE_BRACE_PLACEHOLDER_PATTERN.sub(replace_single_brace, result)

        # Check for missing placeholders in strict mode
        if strict_mode and missing_placeholders:
            unique_missing = list(set(missing_placeholders))
            raise PlaceholderError(f"Missing placeholders in strict mode: {unique_missing}")

        logger.info(f"Placeholder replacement completed: {replacement_count} replacements made")

        # Log missing placeholders in non-strict mode
        if not strict_mode and missing_placeholders:
            unique_missing = list(set(missing_placeholders))
            logger.warning(f"Placeholders not found in data: {unique_missing}")

        return result

    except (PlaceholderError, TemplateError):
        raise

    except Exception as e:
        error_msg = f"Unexpected error during placeholder replacement: {e}"
        logger.error(error_msg)
        raise TemplateError(error_msg)


def validate_template(template_path: str) -> Dict[str, Any]:
    """
    Validate template file and analyze its structure.

    Args:
        template_path: Path to template file

    Returns:
        Dictionary with validation results and analysis
    """
    logger.info(f"Validating template: {template_path}")

    try:
        # Load template
        template_content = load_template(template_path)

        # Find placeholders
        placeholders = find_placeholders(template_content)

        # Basic HTML validation
        has_html_tags = bool(re.search(r'<[^>]+>', template_content))
        has_head_section = '<head>' in template_content.lower()
        has_body_section = '<body>' in template_content.lower()

        # Character encoding detection
        encoding_meta = re.search(r'<meta[^>]*charset[^>]*>', template_content, re.IGNORECASE)

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'analysis': {
                'file_size_bytes': len(template_content.encode('utf-8')),
                'character_count': len(template_content),
                'line_count': template_content.count('\n') + 1,
                'placeholders_found': len(placeholders),
                'placeholder_list': placeholders,
                'has_html_structure': has_html_tags,
                'has_head_section': has_head_section,
                'has_body_section': has_body_section,
                'has_encoding_meta': bool(encoding_meta),
            }
        }

        # Add warnings for potential issues
        if not has_html_tags:
            validation_result['warnings'].append("Template does not contain HTML tags")

        if not has_head_section:
            validation_result['warnings'].append("Template missing <head> section")

        if not has_body_section:
            validation_result['warnings'].append("Template missing <body> section")

        if not encoding_meta:
            validation_result['warnings'].append("Template missing charset meta tag")

        if not placeholders:
            validation_result['warnings'].append("No placeholders found in template")

        logger.info(f"Template validation completed: {len(placeholders)} placeholders found")
        return validation_result

    except (TemplateError, TemplateNotFoundError) as e:
        return {
            'valid': False,
            'errors': [str(e)],
            'warnings': [],
            'analysis': {}
        }

    except Exception as e:
        error_msg = f"Unexpected error during template validation: {e}"
        logger.error(error_msg)
        return {
            'valid': False,
            'errors': [error_msg],
            'warnings': [],
            'analysis': {}
        }


def create_sample_template(output_path: str) -> None:
    """
    Create a sample HTML template with common placeholders.

    Args:
        output_path: Path where to save the sample template

    Raises:
        TemplateError: If template creation fails
    """
    logger.info(f"Creating sample template: {output_path}")

    sample_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><!-- FULL_NAME --> - CV</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c5aa0;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
            margin-bottom: 15px;
        }
        .contact-info {
            font-size: 1.1em;
            margin: 10px 0;
        }
        ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        li {
            margin: 5px 0;
        }
        .experience-item, .education-item, .project-item {
            margin-bottom: 20px;
        }
        .item-title {
            font-weight: bold;
            color: #333;
        }
        .item-subtitle {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><!-- FULL_NAME --></h1>
        <div class="contact-info">
            <div>Email: <!-- EMAIL --></div>
            <div>Phone: <!-- PHONE --></div>
            <div>Location: <!-- LOCATION --></div>
            <div>LinkedIn: <!-- LINKEDIN_URL --></div>
            <div>GitHub: <!-- GITHUB_URL --></div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Professional Summary</div>
        <p><!-- PROFESSIONAL_SUMMARY --></p>
    </div>

    <div class="section">
        <div class="section-title">Technical Skills</div>
        <!-- TECHNICAL_SKILLS -->
    </div>

    <div class="section">
        <div class="section-title">Work Experience</div>
        <!-- WORK_EXPERIENCE -->
    </div>

    <div class="section">
        <div class="section-title">Education</div>
        <!-- EDUCATION -->
    </div>

    <div class="section">
        <div class="section-title">Projects</div>
        <!-- PROJECTS -->
    </div>

    <div class="section">
        <div class="section-title">Additional Skills</div>
        <!-- ADDITIONAL_SKILLS -->
    </div>
</body>
</html>'''

    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(sample_template)

        logger.info(f"Sample template created successfully: {output_path}")

    except Exception as e:
        error_msg = f"Failed to create sample template: {e}"
        logger.error(error_msg)
        raise TemplateError(error_msg)


def preview_template_with_data(template_path: str, data: Dict[str, Any]) -> str:
    """
    Generate a preview of template with provided data.

    Args:
        template_path: Path to template file
        data: Data to use for placeholder replacement

    Returns:
        HTML content with placeholders replaced

    Raises:
        TemplateError: If preview generation fails
    """
    logger.info(f"Generating template preview: {template_path}")

    try:
        # Load template
        template = load_template(template_path)

        # Replace placeholders
        result = replace_placeholders(template, data, strict_mode=False)

        logger.info("Template preview generated successfully")
        return result

    except Exception as e:
        error_msg = f"Failed to generate template preview: {e}"
        logger.error(error_msg)
        raise TemplateError(error_msg)