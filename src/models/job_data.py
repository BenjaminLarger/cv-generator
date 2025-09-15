"""
JobData model for storing and validating job posting information.

This module contains the JobData Pydantic model which represents structured
data about a job posting, including company information, position details,
requirements, and metadata.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum


class ExperienceLevel(str, Enum):
    """Enumeration for experience levels."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"
    INTERN = "intern"


class JobData(BaseModel):
    """
    Represents a job posting with all relevant details.

    This model stores comprehensive information about a job posting including
    company details, position requirements, and extraction metadata. All fields
    are validated for proper formatting and content.

    Attributes:
        company_name: Name of the hiring company
        position: Job title/position name
        requirements: List of job requirements and qualifications
        skills_required: List of technical and soft skills needed
        experience_level: Required experience level (enum)
        description: Full job description text
        url: Optional URL to the original job posting
        extracted_at: Timestamp when the job data was extracted
    """

    company_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the hiring company"
    )

    position: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Job title or position name"
    )

    requirements: List[str] = Field(
        default_factory=list,
        description="List of job requirements and qualifications"
    )

    skills_required: List[str] = Field(
        default_factory=list,
        description="List of technical and soft skills needed"
    )

    experience_level: ExperienceLevel = Field(
        default=ExperienceLevel.MID,
        description="Required experience level"
    )

    description: str = Field(
        ...,
        min_length=10,
        description="Full job description text"
    )

    url: Optional[HttpUrl] = Field(
        default=None,
        description="URL to the original job posting"
    )

    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the job data was extracted"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            HttpUrl: str
        }

    @validator('company_name', 'position')
    def validate_non_empty_string(cls, value: str) -> str:
        """Validate that string fields are not empty after stripping whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or contain only whitespace")
        return cleaned

    @validator('requirements', 'skills_required')
    def validate_string_lists(cls, value: List[str]) -> List[str]:
        """Validate and clean string lists, removing empty items."""
        if not isinstance(value, list):
            raise ValueError("Must be a list of strings")

        cleaned_items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All items must be strings")
            cleaned_item = item.strip()
            if cleaned_item:  # Only add non-empty items
                cleaned_items.append(cleaned_item)

        return cleaned_items

    @validator('description')
    def validate_description(cls, value: str) -> str:
        """Validate job description has meaningful content."""
        cleaned = value.strip()
        if len(cleaned) < 10:
            raise ValueError("Description must be at least 10 characters long")
        return cleaned

    @validator('extracted_at')
    def validate_extracted_at(cls, value: datetime) -> datetime:
        """Ensure extracted_at is not in the future."""
        if value > datetime.now():
            raise ValueError("Extraction timestamp cannot be in the future")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary with proper serialization.

        Returns:
            Dictionary representation of the job data with serialized datetime and URL
        """
        data = self.dict()

        # Convert datetime to ISO string
        if isinstance(data.get('extracted_at'), datetime):
            data['extracted_at'] = data['extracted_at'].isoformat()

        # Convert HttpUrl to string
        if data.get('url'):
            data['url'] = str(data['url'])

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobData':
        """
        Create a JobData instance from a dictionary.

        Args:
            data: Dictionary containing job data

        Returns:
            JobData instance

        Raises:
            ValueError: If data is invalid or missing required fields
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        # Handle datetime conversion
        if 'extracted_at' in data and isinstance(data['extracted_at'], str):
            try:
                data['extracted_at'] = datetime.fromisoformat(data['extracted_at'])
            except ValueError as e:
                raise ValueError(f"Invalid datetime format for extracted_at: {e}")

        return cls(**data)

    def get_summary(self) -> str:
        """
        Generate a concise summary of the job posting.

        Returns:
            A formatted string summarizing the job posting
        """
        skills_summary = ", ".join(self.skills_required[:5])  # First 5 skills
        if len(self.skills_required) > 5:
            skills_summary += f" (and {len(self.skills_required) - 5} more)"

        return (
            f"{self.position} at {self.company_name} "
            f"({self.experience_level.value} level)"
            + (f" - Key skills: {skills_summary}" if skills_summary else "")
        )

    def is_skill_match(self, user_skills: List[str]) -> bool:
        """
        Check if user skills match any of the required skills.

        Args:
            user_skills: List of user's skills

        Returns:
            True if there's at least one skill match, False otherwise
        """
        if not user_skills or not self.skills_required:
            return False

        # Case-insensitive matching
        user_skills_lower = [skill.lower().strip() for skill in user_skills]
        required_skills_lower = [skill.lower().strip() for skill in self.skills_required]

        return any(skill in required_skills_lower for skill in user_skills_lower)

    def get_skill_matches(self, user_skills: List[str]) -> List[str]:
        """
        Get list of matching skills between user and job requirements.

        Args:
            user_skills: List of user's skills

        Returns:
            List of matching skills
        """
        if not user_skills or not self.skills_required:
            return []

        matches = []
        user_skills_lower = [skill.lower().strip() for skill in user_skills]

        for required_skill in self.skills_required:
            required_lower = required_skill.lower().strip()
            if required_lower in user_skills_lower:
                matches.append(required_skill)

        return matches

    def __str__(self) -> str:
        """String representation of the job data."""
        return f"JobData(position='{self.position}', company='{self.company_name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the job data."""
        return (
            f"JobData(company_name='{self.company_name}', "
            f"position='{self.position}', "
            f"experience_level='{self.experience_level.value}', "
            f"skills_count={len(self.skills_required)}, "
            f"requirements_count={len(self.requirements)})"
        )