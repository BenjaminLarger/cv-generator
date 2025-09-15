"""
File and folder management utility for CV/Cover Letter generator.

This module provides comprehensive file and folder management capabilities including
application folder creation, filename generation, and data persistence with proper
error handling and logging.
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, Union
import re
import shutil

from ..models.job_data import JobData
from ..models.match_result import MatchResult
from ..models.user_profile import UserProfile
from .logging_config import get_scraping_logger

logger = get_scraping_logger()


class FileManagerError(Exception):
    """Custom exception for file manager errors."""
    pass


class FolderCreationError(FileManagerError):
    """Exception raised when folder creation fails."""
    pass


class FileOperationError(FileManagerError):
    """Exception raised when file operations fail."""
    pass


def _sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Sanitize string for use as filename or folder name.

    Args:
        name: String to sanitize
        max_length: Maximum length of sanitized name

    Returns:
        Sanitized string safe for filesystem use
    """
    if not name:
        return "unnamed"

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    sanitized = re.sub(r'[^\w\s-]', '', sanitized)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')

    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "unnamed"

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized


def _format_date_for_filename(date_obj: Union[datetime, date, str] = None) -> str:
    """
    Format date for use in filename.

    Args:
        date_obj: Date to format (defaults to current date)

    Returns:
        Date string in YYYY-MM-DD format
    """
    if date_obj is None:
        date_obj = datetime.now()

    if isinstance(date_obj, str):
        try:
            # Try to parse ISO format
            if 'T' in date_obj:
                date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
            else:
                date_obj = datetime.fromisoformat(date_obj)
        except ValueError:
            # Fallback to current date if parsing fails
            date_obj = datetime.now()

    if isinstance(date_obj, datetime):
        return date_obj.strftime('%Y-%m-%d')
    elif isinstance(date_obj, date):
        return date_obj.strftime('%Y-%m-%d')
    else:
        return datetime.now().strftime('%Y-%m-%d')


def create_application_folder(company: str, date: Optional[str] = None, base_path: Optional[str] = None) -> str:
    """
    Create application folder with organized structure.

    Args:
        company: Company name for the application
        date: Date string (defaults to current date)
        base_path: Base path for applications (defaults to ./applications)

    Returns:
        Path to created application folder

    Raises:
        FolderCreationError: If folder creation fails
    """
    logger.info(f"Creating application folder for company: {company}")

    try:
        if not company or not company.strip():
            raise FolderCreationError("Company name cannot be empty")

        # Use provided base path or default
        if base_path is None:
            base_path = Path.cwd() / "applications"
        else:
            base_path = Path(base_path)

        # Ensure base applications directory exists
        base_path.mkdir(parents=True, exist_ok=True)

        # Sanitize company name and format date
        sanitized_company = _sanitize_filename(company.strip())
        formatted_date = _format_date_for_filename(date)

        # Create folder name: company_YYYY-MM-DD
        folder_name = f"{sanitized_company}_{formatted_date}"

        # Handle duplicate folder names
        application_folder = base_path / folder_name
        counter = 1
        original_folder_name = folder_name

        while application_folder.exists():
            folder_name = f"{original_folder_name}_{counter:02d}"
            application_folder = base_path / folder_name
            counter += 1

        # Create the application folder
        application_folder.mkdir(parents=True)

        # Create subdirectories for organization
        subdirs = ['documents', 'analysis', 'templates', 'resources']
        for subdir in subdirs:
            (application_folder / subdir).mkdir(exist_ok=True)

        # Create a metadata file
        metadata = {
            'company': company,
            'created_date': datetime.now().isoformat(),
            'folder_name': folder_name,
            'application_date': formatted_date
        }

        metadata_path = application_folder / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        folder_path = str(application_folder.absolute())
        logger.info(f"Application folder created successfully: {folder_path}")
        return folder_path

    except PermissionError:
        error_msg = f"Permission denied creating application folder for {company}"
        logger.error(error_msg)
        raise FolderCreationError(error_msg)

    except OSError as e:
        error_msg = f"OS error creating application folder: {e}"
        logger.error(error_msg)
        raise FolderCreationError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error creating application folder: {e}"
        logger.error(error_msg)
        raise FolderCreationError(error_msg)


def generate_filename(doc_type: str, company: str, date: Optional[str] = None, extension: str = None) -> str:
    """
    Generate standardized filename for documents.

    Args:
        doc_type: Type of document (cv, cover_letter, analysis, etc.)
        company: Company name
        date: Date string (defaults to current date)
        extension: File extension (without dot)

    Returns:
        Generated filename string

    Raises:
        FileManagerError: If filename generation fails
    """
    logger.debug(f"Generating filename for {doc_type} - {company}")

    try:
        if not doc_type or not doc_type.strip():
            raise FileManagerError("Document type cannot be empty")

        if not company or not company.strip():
            raise FileManagerError("Company name cannot be empty")

        # Sanitize inputs
        sanitized_doc_type = _sanitize_filename(doc_type.strip().lower())
        sanitized_company = _sanitize_filename(company.strip())
        formatted_date = _format_date_for_filename(date)

        # Generate base filename
        filename_base = f"{sanitized_doc_type}_{sanitized_company}_{formatted_date}"

        # Add extension if provided
        if extension:
            extension = extension.strip('.')
            filename = f"{filename_base}.{extension}"
        else:
            filename = filename_base

        logger.debug(f"Generated filename: {filename}")
        return filename

    except Exception as e:
        error_msg = f"Error generating filename: {e}"
        logger.error(error_msg)
        raise FileManagerError(error_msg)


def save_job_analysis(job_data: JobData, folder: str, filename: Optional[str] = None) -> str:
    """
    Save job analysis data to application folder.

    Args:
        job_data: JobData instance to save
        folder: Application folder path
        filename: Optional custom filename

    Returns:
        Path to saved file

    Raises:
        FileOperationError: If save operation fails
    """
    logger.info(f"Saving job analysis to folder: {folder}")

    try:
        if not isinstance(job_data, JobData):
            raise FileOperationError("job_data must be a JobData instance")

        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileOperationError(f"Folder does not exist: {folder}")

        # Create analysis subfolder if it doesn't exist
        analysis_folder = folder_path / 'analysis'
        analysis_folder.mkdir(exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            filename = generate_filename(
                'job_analysis',
                job_data.company_name,
                extension='json'
            )

        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        file_path = analysis_folder / filename

        # Convert job data to dictionary
        job_dict = job_data.to_dict()

        # Add metadata
        job_dict['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'file_type': 'job_analysis',
            'version': '1.0'
        }

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(job_dict, f, indent=2, ensure_ascii=False)

        saved_path = str(file_path.absolute())
        logger.info(f"Job analysis saved successfully: {saved_path}")
        return saved_path

    except (TypeError, ValueError) as e:
        error_msg = f"Data serialization error: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except PermissionError:
        error_msg = f"Permission denied writing to folder: {folder}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error saving job analysis: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


def save_match_report(match_result: MatchResult, folder: str, filename: Optional[str] = None) -> str:
    """
    Save match analysis report to application folder.

    Args:
        match_result: MatchResult instance to save
        folder: Application folder path
        filename: Optional custom filename

    Returns:
        Path to saved file

    Raises:
        FileOperationError: If save operation fails
    """
    logger.info(f"Saving match report to folder: {folder}")

    try:
        if not isinstance(match_result, MatchResult):
            raise FileOperationError("match_result must be a MatchResult instance")

        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileOperationError(f"Folder does not exist: {folder}")

        # Create analysis subfolder if it doesn't exist
        analysis_folder = folder_path / 'analysis'
        analysis_folder.mkdir(exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            filename = generate_filename(
                'match_report',
                match_result.company_name,
                extension='json'
            )

        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        file_path = analysis_folder / filename

        # Convert match result to dictionary
        match_dict = match_result.to_dict()

        # Add metadata
        match_dict['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'file_type': 'match_report',
            'version': '1.0'
        }

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(match_dict, f, indent=2, ensure_ascii=False)

        saved_path = str(file_path.absolute())
        logger.info(f"Match report saved successfully: {saved_path}")
        return saved_path

    except (TypeError, ValueError) as e:
        error_msg = f"Data serialization error: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except PermissionError:
        error_msg = f"Permission denied writing to folder: {folder}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error saving match report: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


def save_user_profile_copy(user_profile: UserProfile, folder: str, filename: Optional[str] = None) -> str:
    """
    Save a copy of user profile to application folder.

    Args:
        user_profile: UserProfile instance to save
        folder: Application folder path
        filename: Optional custom filename

    Returns:
        Path to saved file

    Raises:
        FileOperationError: If save operation fails
    """
    logger.info(f"Saving user profile copy to folder: {folder}")

    try:
        if not isinstance(user_profile, UserProfile):
            raise FileOperationError("user_profile must be a UserProfile instance")

        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileOperationError(f"Folder does not exist: {folder}")

        # Create resources subfolder if it doesn't exist
        resources_folder = folder_path / 'resources'
        resources_folder.mkdir(exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            filename = generate_filename(
                'user_profile',
                user_profile.personal_info.name,
                extension='json'
            )

        # Ensure filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'

        file_path = resources_folder / filename

        # Convert user profile to dictionary
        profile_dict = user_profile.to_dict()

        # Add metadata
        profile_dict['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'file_type': 'user_profile',
            'version': '1.0'
        }

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)

        saved_path = str(file_path.absolute())
        logger.info(f"User profile copy saved successfully: {saved_path}")
        return saved_path

    except (TypeError, ValueError) as e:
        error_msg = f"Data serialization error: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except PermissionError:
        error_msg = f"Permission denied writing to folder: {folder}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error saving user profile: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


def copy_template_to_folder(template_path: str, folder: str, new_name: Optional[str] = None) -> str:
    """
    Copy template file to application folder.

    Args:
        template_path: Source template file path
        folder: Application folder path
        new_name: Optional new name for the template

    Returns:
        Path to copied template

    Raises:
        FileOperationError: If copy operation fails
    """
    logger.info(f"Copying template {template_path} to folder: {folder}")

    try:
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileOperationError(f"Template file not found: {template_path}")

        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileOperationError(f"Folder does not exist: {folder}")

        # Create templates subfolder if it doesn't exist
        templates_folder = folder_path / 'templates'
        templates_folder.mkdir(exist_ok=True)

        # Determine destination filename
        if new_name:
            destination_name = new_name
            if not destination_name.endswith(template_file.suffix):
                destination_name += template_file.suffix
        else:
            destination_name = template_file.name

        destination_path = templates_folder / destination_name

        # Copy file
        shutil.copy2(template_file, destination_path)

        copied_path = str(destination_path.absolute())
        logger.info(f"Template copied successfully: {copied_path}")
        return copied_path

    except PermissionError:
        error_msg = f"Permission denied copying template: {template_path}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except OSError as e:
        error_msg = f"OS error copying template: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error copying template: {e}"
        logger.error(error_msg)
        raise FileOperationError(error_msg)


def get_application_folders(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get list of all application folders with metadata.

    Args:
        base_path: Base path to search (defaults to ./applications)

    Returns:
        List of dictionaries containing folder information
    """
    logger.info("Getting list of application folders")

    try:
        if base_path is None:
            base_path = Path.cwd() / "applications"
        else:
            base_path = Path(base_path)

        if not base_path.exists():
            return []

        folders = []
        for folder_path in base_path.iterdir():
            if folder_path.is_dir():
                folder_info = {
                    'path': str(folder_path.absolute()),
                    'name': folder_path.name,
                    'created': datetime.fromtimestamp(folder_path.stat().st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(folder_path.stat().st_mtime).isoformat(),
                }

                # Try to read metadata if available
                metadata_file = folder_path / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        folder_info['metadata'] = metadata
                    except Exception:
                        logger.warning(f"Failed to read metadata for folder: {folder_path}")

                folders.append(folder_info)

        # Sort by creation date (newest first)
        folders.sort(key=lambda x: x['created'], reverse=True)

        logger.info(f"Found {len(folders)} application folders")
        return folders

    except Exception as e:
        logger.warning(f"Error getting application folders: {e}")
        return []


def cleanup_old_applications(base_path: Optional[str] = None, days_old: int = 90) -> int:
    """
    Clean up application folders older than specified days.

    Args:
        base_path: Base path to search (defaults to ./applications)
        days_old: Number of days after which folders are considered old

    Returns:
        Number of folders cleaned up
    """
    logger.info(f"Cleaning up applications older than {days_old} days")

    try:
        if base_path is None:
            base_path = Path.cwd() / "applications"
        else:
            base_path = Path(base_path)

        if not base_path.exists():
            return 0

        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        cleaned_count = 0

        for folder_path in base_path.iterdir():
            if folder_path.is_dir():
                if folder_path.stat().st_mtime < cutoff_time:
                    try:
                        shutil.rmtree(folder_path)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old folder: {folder_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up folder {folder_path.name}: {e}")

        logger.info(f"Cleanup completed: {cleaned_count} folders removed")
        return cleaned_count

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return 0