"""
UserProfile model for storing and validating user profile information.

This module contains Pydantic models for representing a comprehensive user profile
including personal information, work experience, education, skills, and projects.
All models include proper validation and serialization capabilities.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, EmailStr, HttpUrl
from enum import Enum
import re


class PersonalInfo(BaseModel):
    """Personal information model for user profile."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Full name of the user"
    )

    email: EmailStr = Field(
        ...,
        description="Email address"
    )

    phone: Optional[str] = Field(
        default=None,
        description="Phone number"
    )

    location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Current location (city, country)"
    )

    @validator('name')
    def validate_name(cls, value: str) -> str:
        """Validate and clean name field."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Name cannot be empty")

        # Check for at least one letter
        if not re.search(r'[a-zA-Z]', cleaned):
            raise ValueError("Name must contain at least one letter")

        return cleaned

    @validator('phone')
    def validate_phone(cls, value: Optional[str]) -> Optional[str]:
        """Validate phone number format."""
        if value is None:
            return value

        cleaned = re.sub(r'[^\d+\-\(\)\s]', '', value.strip())
        if not cleaned:
            return None

        # Basic phone validation - must contain digits
        if not re.search(r'\d', cleaned):
            raise ValueError("Phone number must contain digits")

        return cleaned


class WorkExperience(BaseModel):
    """Work experience entry model."""

    company: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Company name"
    )

    role: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Job title/role"
    )

    start_date: date = Field(
        ...,
        description="Start date of employment"
    )

    end_date: Optional[date] = Field(
        default=None,
        description="End date of employment (None if current)"
    )

    duration: Optional[str] = Field(
        default=None,
        description="Duration string (e.g., '2 years 3 months')"
    )

    achievements: List[str] = Field(
        default_factory=list,
        description="List of achievements and responsibilities"
    )

    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies and tools used"
    )

    location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Job location"
    )

    @validator('company', 'role')
    def validate_required_strings(cls, value: str) -> str:
        """Validate required string fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty")
        return cleaned

    @validator('end_date')
    def validate_end_date(cls, value: Optional[date], values: Dict[str, Any]) -> Optional[date]:
        """Validate that end_date is after start_date."""
        if value is not None and 'start_date' in values:
            start_date = values['start_date']
            if value < start_date:
                raise ValueError("End date cannot be before start date")
        return value

    @validator('achievements', 'technologies')
    def validate_string_lists(cls, value: List[str]) -> List[str]:
        """Validate and clean string lists."""
        cleaned_items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All items must be strings")
            cleaned_item = item.strip()
            if cleaned_item:
                cleaned_items.append(cleaned_item)
        return cleaned_items

    def is_current_job(self) -> bool:
        """Check if this is a current job (no end date)."""
        return self.end_date is None

    def get_duration_months(self) -> Optional[int]:
        """Calculate duration in months."""
        if not self.start_date:
            return None

        end = self.end_date or date.today()
        years = end.year - self.start_date.year
        months = end.month - self.start_date.month

        return years * 12 + months


class Education(BaseModel):
    """Education entry model."""

    degree: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Degree name (e.g., Bachelor of Science)"
    )

    field_of_study: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Field of study or major"
    )

    institution: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Educational institution name"
    )

    graduation_year: Optional[int] = Field(
        default=None,
        ge=1950,
        le=2050,
        description="Year of graduation"
    )

    gpa: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=4.0,
        description="GPA (0.0-4.0 scale)"
    )

    honors: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Academic honors or distinctions"
    )

    location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Institution location"
    )

    @validator('degree', 'institution')
    def validate_required_strings(cls, value: str) -> str:
        """Validate required string fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty")
        return cleaned

    @validator('graduation_year')
    def validate_graduation_year(cls, value: Optional[int]) -> Optional[int]:
        """Validate graduation year is reasonable."""
        if value is not None:
            current_year = datetime.now().year
            if value > current_year + 10:
                raise ValueError("Graduation year cannot be more than 10 years in the future")
        return value


class Project(BaseModel):
    """Project entry model."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Project title"
    )

    description: str = Field(
        ...,
        min_length=10,
        description="Project description"
    )

    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies and tools used"
    )

    url: Optional[HttpUrl] = Field(
        default=None,
        description="Project URL (demo, repository, etc.)"
    )

    start_date: Optional[date] = Field(
        default=None,
        description="Project start date"
    )

    end_date: Optional[date] = Field(
        default=None,
        description="Project completion date"
    )

    status: str = Field(
        default="completed",
        description="Project status (completed, ongoing, paused)"
    )

    @validator('title')
    def validate_title(cls, value: str) -> str:
        """Validate project title."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Project title cannot be empty")
        return cleaned

    @validator('description')
    def validate_description(cls, value: str) -> str:
        """Validate project description."""
        cleaned = value.strip()
        if len(cleaned) < 10:
            raise ValueError("Project description must be at least 10 characters")
        return cleaned

    @validator('technologies')
    def validate_technologies(cls, value: List[str]) -> List[str]:
        """Validate technologies list."""
        cleaned_items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All technologies must be strings")
            cleaned_item = item.strip()
            if cleaned_item:
                cleaned_items.append(cleaned_item)
        return cleaned_items

    @validator('status')
    def validate_status(cls, value: str) -> str:
        """Validate project status."""
        valid_statuses = {'completed', 'ongoing', 'paused', 'archived'}
        if value.lower() not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return value.lower()


class SocialUrls(BaseModel):
    """Social media and professional URLs model."""

    github: Optional[HttpUrl] = Field(
        default=None,
        description="GitHub profile URL"
    )

    linkedin: Optional[HttpUrl] = Field(
        default=None,
        description="LinkedIn profile URL"
    )

    portfolio: Optional[HttpUrl] = Field(
        default=None,
        description="Personal portfolio website URL"
    )

    twitter: Optional[HttpUrl] = Field(
        default=None,
        description="Twitter profile URL"
    )

    website: Optional[HttpUrl] = Field(
        default=None,
        description="Personal website URL"
    )

    other: List[HttpUrl] = Field(
        default_factory=list,
        description="Other relevant URLs"
    )


class UserProfile(BaseModel):
    """
    Comprehensive user profile model containing all user information.

    This model represents a complete user profile including personal information,
    work experience, education, skills, projects, and social media links.
    All nested models provide comprehensive validation and serialization.

    Attributes:
        personal_info: Personal contact and basic information
        experiences: List of work experiences
        skills: List of user's skills and competencies
        education: List of educational qualifications
        projects: List of personal/professional projects
        urls: Social media and professional profile URLs
        created_at: Profile creation timestamp
        updated_at: Last update timestamp
    """

    personal_info: PersonalInfo = Field(
        ...,
        description="Personal contact and basic information"
    )

    experiences: List[WorkExperience] = Field(
        default_factory=list,
        description="List of work experiences"
    )

    skills: List[str] = Field(
        default_factory=list,
        description="List of skills and competencies"
    )

    education: List[Education] = Field(
        default_factory=list,
        description="List of educational qualifications"
    )

    projects: List[Project] = Field(
        default_factory=list,
        description="List of personal/professional projects"
    )

    urls: SocialUrls = Field(
        default_factory=SocialUrls,
        description="Social media and professional profile URLs"
    )

    summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Professional summary or bio"
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Profile creation timestamp"
    )

    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            HttpUrl: str
        }

    @validator('skills')
    def validate_skills(cls, value: List[str]) -> List[str]:
        """Validate and clean skills list."""
        cleaned_skills = []
        seen_skills = set()

        for skill in value:
            if not isinstance(skill, str):
                raise ValueError("All skills must be strings")

            cleaned_skill = skill.strip().title()
            if cleaned_skill and cleaned_skill.lower() not in seen_skills:
                cleaned_skills.append(cleaned_skill)
                seen_skills.add(cleaned_skill.lower())

        return cleaned_skills

    @validator('experiences')
    def validate_experiences(cls, value: List[WorkExperience]) -> List[WorkExperience]:
        """Validate work experiences list."""
        if not isinstance(value, list):
            raise ValueError("Experiences must be a list")

        # Sort by start date (most recent first)
        return sorted(value, key=lambda x: x.start_date, reverse=True)

    @validator('education')
    def validate_education(cls, value: List[Education]) -> List[Education]:
        """Validate education list."""
        if not isinstance(value, list):
            raise ValueError("Education must be a list")

        # Sort by graduation year (most recent first)
        return sorted(value,
                     key=lambda x: x.graduation_year or 0,
                     reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary with proper serialization.

        Returns:
            Dictionary representation of the user profile
        """
        data = self.dict()

        # Convert datetime fields to ISO strings
        for field in ['created_at', 'updated_at']:
            if isinstance(data.get(field), datetime):
                data[field] = data[field].isoformat()

        # Handle nested models serialization
        for experience in data.get('experiences', []):
            if 'start_date' in experience and experience['start_date']:
                experience['start_date'] = experience['start_date'].isoformat()
            if 'end_date' in experience and experience['end_date']:
                experience['end_date'] = experience['end_date'].isoformat()

        for education in data.get('education', []):
            # Education dates are already handled by the model
            pass

        for project in data.get('projects', []):
            if 'start_date' in project and project['start_date']:
                project['start_date'] = project['start_date'].isoformat()
            if 'end_date' in project and project['end_date']:
                project['end_date'] = project['end_date'].isoformat()
            if 'url' in project and project['url']:
                project['url'] = str(project['url'])

        # Convert URLs to strings
        for url_field in data.get('urls', {}):
            if data['urls'][url_field]:
                if isinstance(data['urls'][url_field], list):
                    data['urls'][url_field] = [str(url) for url in data['urls'][url_field]]
                else:
                    data['urls'][url_field] = str(data['urls'][url_field])

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """
        Create a UserProfile instance from a dictionary.

        Args:
            data: Dictionary containing user profile data

        Returns:
            UserProfile instance

        Raises:
            ValueError: If data is invalid or missing required fields
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        # Handle datetime conversion
        for field in ['created_at', 'updated_at']:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except ValueError as e:
                    raise ValueError(f"Invalid datetime format for {field}: {e}")

        # Handle nested date conversions
        for experience in data.get('experiences', []):
            for date_field in ['start_date', 'end_date']:
                if date_field in experience and isinstance(experience[date_field], str):
                    try:
                        experience[date_field] = date.fromisoformat(experience[date_field])
                    except ValueError as e:
                        raise ValueError(f"Invalid date format for experience {date_field}: {e}")

        for project in data.get('projects', []):
            for date_field in ['start_date', 'end_date']:
                if date_field in project and isinstance(project[date_field], str):
                    try:
                        project[date_field] = date.fromisoformat(project[date_field])
                    except ValueError as e:
                        raise ValueError(f"Invalid date format for project {date_field}: {e}")

        return cls(**data)

    def get_total_experience_years(self) -> float:
        """
        Calculate total years of work experience.

        Returns:
            Total years of experience as a float
        """
        total_months = 0
        for exp in self.experiences:
            months = exp.get_duration_months()
            if months:
                total_months += months

        return round(total_months / 12, 1)

    def get_current_position(self) -> Optional[str]:
        """
        Get current job position if available.

        Returns:
            Current position string or None
        """
        for exp in self.experiences:
            if exp.is_current_job():
                return f"{exp.role} at {exp.company}"
        return None

    def get_skills_by_category(self, tech_keywords: List[str] = None) -> Dict[str, List[str]]:
        """
        Categorize skills into technical and soft skills.

        Args:
            tech_keywords: List of keywords to identify technical skills

        Returns:
            Dictionary with 'technical' and 'soft' skill lists
        """
        if tech_keywords is None:
            tech_keywords = [
                'python', 'java', 'javascript', 'react', 'angular', 'vue',
                'sql', 'mongodb', 'postgresql', 'mysql', 'redis',
                'aws', 'azure', 'gcp', 'docker', 'kubernetes',
                'git', 'jenkins', 'ci/cd', 'agile', 'scrum'
            ]

        technical = []
        soft = []

        for skill in self.skills:
            is_technical = any(keyword.lower() in skill.lower() for keyword in tech_keywords)
            if is_technical:
                technical.append(skill)
            else:
                soft.append(skill)

        return {'technical': technical, 'soft': soft}

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = datetime.now()

    def __str__(self) -> str:
        """String representation of the user profile."""
        return f"UserProfile(name='{self.personal_info.name}', email='{self.personal_info.email}')"

    def __repr__(self) -> str:
        """Detailed string representation of the user profile."""
        return (
            f"UserProfile(name='{self.personal_info.name}', "
            f"experiences_count={len(self.experiences)}, "
            f"skills_count={len(self.skills)}, "
            f"education_count={len(self.education)}, "
            f"projects_count={len(self.projects)})"
        )