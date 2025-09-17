"""
YAML parsing utility for user profile management.

This module provides robust YAML parsing and validation capabilities for loading
and saving user profiles with comprehensive error handling and schema validation.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, date

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.user_profile import UserProfile, PersonalInfo, WorkExperience, Education, Project, SocialUrls
from utils.logging_config import get_scraping_logger

logger = get_scraping_logger()


class YAMLParser:
    """YAML parser class for convenient access to parsing functions."""

    @staticmethod
    def load_user_profile(yaml_path: str) -> UserProfile:
        """Load user profile from YAML file."""
        return load_user_profile(yaml_path)

    @staticmethod
    def save_user_profile(profile: UserProfile, path: str) -> None:
        """Save user profile to YAML file."""
        return save_user_profile(profile, path)

    @staticmethod
    def validate_yaml_file(yaml_path: str) -> Dict[str, Any]:
        """Validate YAML file."""
        return validate_yaml_file(yaml_path)

    @staticmethod
    def create_sample_yaml(output_path: str) -> None:
        """Create sample YAML file."""
        return create_sample_yaml(output_path)


class YamlParsingError(Exception):
    """Custom exception for YAML parsing errors."""
    pass


class YamlValidationError(Exception):
    """Custom exception for YAML validation errors."""
    pass


def _convert_date_strings(data: Dict[str, Any], date_fields: List[str]) -> Dict[str, Any]:
    """
    Convert date strings to date objects in nested data structures.

    Args:
        data: Dictionary potentially containing date strings
        date_fields: List of field names that should be date objects

    Returns:
        Dictionary with converted dates
    """
    converted_data = data.copy()

    for field in date_fields:
        if field in converted_data and converted_data[field]:
            if isinstance(converted_data[field], str):
                try:
                    converted_data[field] = date.fromisoformat(converted_data[field])
                except ValueError as e:
                    logger.warning(f"Invalid date format for {field}: {converted_data[field]}")
                    raise YamlValidationError(f"Invalid date format for {field}: {e}")

    return converted_data


def _convert_datetime_strings(data: Dict[str, Any], datetime_fields: List[str]) -> Dict[str, Any]:
    """
    Convert datetime strings to datetime objects in nested data structures.

    Args:
        data: Dictionary potentially containing datetime strings
        datetime_fields: List of field names that should be datetime objects

    Returns:
        Dictionary with converted datetimes
    """
    converted_data = data.copy()

    for field in datetime_fields:
        if field in converted_data and converted_data[field]:
            if isinstance(converted_data[field], str):
                try:
                    converted_data[field] = datetime.fromisoformat(converted_data[field])
                except ValueError as e:
                    logger.warning(f"Invalid datetime format for {field}: {converted_data[field]}")
                    raise YamlValidationError(f"Invalid datetime format for {field}: {e}")

    return converted_data


def validate_yaml_schema(data: Dict[str, Any]) -> bool:
    """
    Validate YAML data against UserProfile schema requirements.

    Args:
        data: Dictionary containing YAML data

    Returns:
        True if schema is valid

    Raises:
        YamlValidationError: If schema validation fails
    """
    logger.debug("Starting YAML schema validation")

    try:
        # Check required top-level structure
        if not isinstance(data, dict):
            raise YamlValidationError("YAML data must be a dictionary")

        # Check for required personal_info section
        if 'personal_info' not in data:
            raise YamlValidationError("Missing required 'personal_info' section")

        personal_info = data['personal_info']
        if not isinstance(personal_info, dict):
            raise YamlValidationError("'personal_info' must be a dictionary")

        # Check required personal_info fields
        required_personal_fields = ['name', 'email']
        for field in required_personal_fields:
            if field not in personal_info or not personal_info[field]:
                raise YamlValidationError(f"Missing required personal_info field: {field}")

        # Validate experiences structure if present
        if 'experiences' in data and data['experiences']:
            if not isinstance(data['experiences'], list):
                raise YamlValidationError("'experiences' must be a list")

            for i, exp in enumerate(data['experiences']):
                if not isinstance(exp, dict):
                    raise YamlValidationError(f"Experience {i} must be a dictionary")

                required_exp_fields = ['company', 'role', 'start_date']
                for field in required_exp_fields:
                    if field not in exp or not exp[field]:
                        raise YamlValidationError(f"Missing required experience field: {field} (experience {i})")

        # Validate education structure if present
        if 'education' in data and data['education']:
            if not isinstance(data['education'], list):
                raise YamlValidationError("'education' must be a list")

            for i, edu in enumerate(data['education']):
                if not isinstance(edu, dict):
                    raise YamlValidationError(f"Education {i} must be a dictionary")

                required_edu_fields = ['degree', 'institution']
                for field in required_edu_fields:
                    if field not in edu or not edu[field]:
                        raise YamlValidationError(f"Missing required education field: {field} (education {i})")

        # Validate projects structure if present
        if 'projects' in data and data['projects']:
            if not isinstance(data['projects'], list):
                raise YamlValidationError("'projects' must be a list")

            for i, project in enumerate(data['projects']):
                if not isinstance(project, dict):
                    raise YamlValidationError(f"Project {i} must be a dictionary")

                required_project_fields = ['title', 'description']
                for field in required_project_fields:
                    if field not in project or not project[field]:
                        raise YamlValidationError(f"Missing required project field: {field} (project {i})")

        # Validate skills structure if present
        if 'skills' in data and data['skills']:
            if not isinstance(data['skills'], list):
                raise YamlValidationError("'skills' must be a list")

            for i, skill in enumerate(data['skills']):
                if not isinstance(skill, str):
                    raise YamlValidationError(f"Skill {i} must be a string")

        # Validate urls structure if present
        if 'urls' in data and data['urls']:
            if not isinstance(data['urls'], dict):
                raise YamlValidationError("'urls' must be a dictionary")

        logger.debug("YAML schema validation passed")
        return True

    except YamlValidationError:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during schema validation: {e}"
        logger.error(error_msg)
        raise YamlValidationError(error_msg)


def load_user_profile(yaml_path: str) -> UserProfile:
    """
    Load and validate user profile from YAML file.

    Args:
        yaml_path: Path to the YAML file containing user profile

    Returns:
        UserProfile instance

    Raises:
        YamlParsingError: If file reading or YAML parsing fails
        YamlValidationError: If data validation fails
    """
    logger.info(f"Loading user profile from: {yaml_path}")

    try:
        yaml_file = Path(yaml_path)

        # Check file exists
        if not yaml_file.exists():
            raise YamlParsingError(f"YAML file not found: {yaml_path}")

        if not yaml_file.is_file():
            raise YamlParsingError(f"Path is not a file: {yaml_path}")

        # Read and parse YAML
        with open(yaml_file, 'r', encoding='utf-8') as file:
            raw_data = yaml.safe_load(file)

        if raw_data is None:
            raise YamlParsingError("YAML file is empty or contains no data")

        # Validate schema
        validate_yaml_schema(raw_data)

        # Process the data for UserProfile creation
        processed_data = _process_yaml_data(raw_data)

        # Create UserProfile instance
        profile = UserProfile(**processed_data)

        logger.info(f"Successfully loaded user profile for: {profile.personal_info.name}")
        return profile

    except yaml.YAMLError as e:
        error_msg = f"YAML parsing error in {yaml_path}: {e}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)

    except FileNotFoundError:
        error_msg = f"YAML file not found: {yaml_path}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)

    except PermissionError:
        error_msg = f"Permission denied reading YAML file: {yaml_path}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)

    except (YamlValidationError, ValueError) as e:
        logger.error(f"Data validation error: {e}")
        raise

    except Exception as e:
        error_msg = f"Unexpected error loading user profile: {e}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)


def _process_yaml_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw YAML data into format suitable for UserProfile creation.

    Args:
        data: Raw YAML data dictionary

    Returns:
        Processed data dictionary
    """
    processed = data.copy()

    # Convert date strings in experiences
    if 'experiences' in processed and processed['experiences']:
        for exp in processed['experiences']:
            exp = _convert_date_strings(exp, ['start_date', 'end_date'])

    # Convert date strings in projects
    if 'projects' in processed and processed['projects']:
        for project in processed['projects']:
            project = _convert_date_strings(project, ['start_date', 'end_date'])

    # Convert datetime strings at profile level
    processed = _convert_datetime_strings(processed, ['created_at', 'updated_at'])

    # Ensure required nested structures exist
    if 'urls' not in processed:
        processed['urls'] = {}

    if 'experiences' not in processed:
        processed['experiences'] = []

    if 'education' not in processed:
        processed['education'] = []

    if 'projects' not in processed:
        processed['projects'] = []

    if 'skills' not in processed:
        processed['skills'] = []

    return processed


def save_user_profile(profile: UserProfile, path: str) -> None:
    """
    Save user profile to YAML file.

    Args:
        profile: UserProfile instance to save
        path: Output YAML file path

    Raises:
        YamlParsingError: If file writing fails
    """
    logger.info(f"Saving user profile to: {path}")

    try:
        output_path = Path(path)

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert profile to dictionary
        profile_data = profile.to_dict()

        # Write YAML file
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(
                profile_data,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )

        logger.info(f"Successfully saved user profile for: {profile.personal_info.name}")

    except PermissionError:
        error_msg = f"Permission denied writing to: {path}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)

    except OSError as e:
        error_msg = f"OS error writing YAML file: {e}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error saving user profile: {e}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)


def create_sample_yaml(output_path: str) -> None:
    """
    Create a sample YAML file with proper structure for reference.

    Args:
        output_path: Path where to save the sample YAML file

    Raises:
        YamlParsingError: If file creation fails
    """
    logger.info(f"Creating sample YAML file: {output_path}")

    sample_data = {
        'personal_info': {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '+1-555-123-4567',
            'location': 'San Francisco, CA'
        },
        'summary': 'Experienced software engineer with expertise in Python, web development, and cloud technologies.',
        'skills': [
            'Python',
            'JavaScript',
            'React',
            'Node.js',
            'AWS',
            'Docker',
            'PostgreSQL',
            'Git'
        ],
        'experiences': [
            {
                'company': 'Tech Corp',
                'role': 'Senior Software Engineer',
                'start_date': '2022-01-15',
                'end_date': None,  # Current job
                'location': 'San Francisco, CA',
                'achievements': [
                    'Led development of microservices architecture serving 1M+ users',
                    'Reduced API response time by 40% through optimization',
                    'Mentored 3 junior developers'
                ],
                'technologies': ['Python', 'FastAPI', 'PostgreSQL', 'AWS', 'Docker']
            },
            {
                'company': 'Startup Inc',
                'role': 'Full Stack Developer',
                'start_date': '2020-06-01',
                'end_date': '2021-12-31',
                'location': 'Remote',
                'achievements': [
                    'Built responsive web application from scratch',
                    'Implemented CI/CD pipeline reducing deployment time by 60%'
                ],
                'technologies': ['JavaScript', 'React', 'Node.js', 'MongoDB']
            }
        ],
        'education': [
            {
                'degree': 'Bachelor of Science',
                'field_of_study': 'Computer Science',
                'institution': 'University of California, Berkeley',
                'graduation_year': 2020,
                'gpa': 3.8,
                'location': 'Berkeley, CA'
            }
        ],
        'projects': [
            {
                'title': 'E-commerce Platform',
                'description': 'Full-stack e-commerce platform with payment integration and inventory management',
                'technologies': ['Python', 'Django', 'React', 'PostgreSQL', 'Stripe API'],
                'url': 'https://github.com/johndoe/ecommerce-platform',
                'status': 'completed',
                'start_date': '2021-03-01',
                'end_date': '2021-08-15'
            },
            {
                'title': 'Weather Forecast App',
                'description': 'Mobile-responsive weather application with location-based forecasts',
                'technologies': ['JavaScript', 'React Native', 'OpenWeather API'],
                'url': 'https://weather-app.johndoe.com',
                'status': 'ongoing'
            }
        ],
        'urls': {
            'github': 'https://github.com/johndoe',
            'linkedin': 'https://linkedin.com/in/johndoe',
            'portfolio': 'https://johndoe.dev',
            'website': 'https://johndoe.com'
        }
    }

    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write("# User Profile YAML Template\n")
            file.write("# This file contains a sample user profile structure\n")
            file.write("# Modify the values below to match your information\n\n")

            yaml.safe_dump(
                sample_data,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )

        logger.info(f"Sample YAML file created successfully: {output_path}")

    except Exception as e:
        error_msg = f"Failed to create sample YAML file: {e}"
        logger.error(error_msg)
        raise YamlParsingError(error_msg)


def validate_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """
    Validate YAML file without creating UserProfile instance.

    Args:
        yaml_path: Path to YAML file to validate

    Returns:
        Dictionary with validation results

    Raises:
        YamlParsingError: If file reading or parsing fails
    """
    logger.info(f"Validating YAML file: {yaml_path}")

    try:
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise YamlParsingError(f"YAML file not found: {yaml_path}")

        with open(yaml_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        if data is None:
            raise YamlParsingError("YAML file is empty or contains no data")

        # Validate schema
        validate_yaml_schema(data)

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {
                'has_experiences': bool(data.get('experiences')),
                'experiences_count': len(data.get('experiences', [])),
                'has_education': bool(data.get('education')),
                'education_count': len(data.get('education', [])),
                'has_projects': bool(data.get('projects')),
                'projects_count': len(data.get('projects', [])),
                'has_skills': bool(data.get('skills')),
                'skills_count': len(data.get('skills', [])),
            }
        }

        logger.info(f"YAML validation successful for: {yaml_path}")
        return validation_result

    except (YamlParsingError, YamlValidationError) as e:
        return {
            'valid': False,
            'errors': [str(e)],
            'warnings': [],
            'summary': {}
        }

    except Exception as e:
        error_msg = f"Unexpected error during validation: {e}"
        logger.error(error_msg)
        return {
            'valid': False,
            'errors': [error_msg],
            'warnings': [],
            'summary': {}
        }