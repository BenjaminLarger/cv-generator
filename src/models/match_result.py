"""
MatchResult model for storing job-profile matching analysis results.

This module contains Pydantic models for representing the results of matching
a user profile against a job posting, including match scores, relevant experiences,
and recommendations for improving the match.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class MatchScore(int, Enum):
    """Enumeration for standardized match scores."""
    POOR = 1
    VERY_LOW = 2
    LOW = 3
    BELOW_AVERAGE = 4
    AVERAGE = 5
    ABOVE_AVERAGE = 6
    GOOD = 7
    VERY_GOOD = 8
    EXCELLENT = 9
    PERFECT = 10


class SuggestionCategory(str, Enum):
    """Categories for improvement suggestions."""
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    GENERAL = "general"


class RelevantExperience(BaseModel):
    """Model for relevant work experience in match context."""

    company: str = Field(
        ...,
        min_length=1,
        description="Company name"
    )

    role: str = Field(
        ...,
        min_length=1,
        description="Job role/title"
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0-1.0)"
    )

    matching_skills: List[str] = Field(
        default_factory=list,
        description="Skills that match job requirements"
    )

    key_achievements: List[str] = Field(
        default_factory=list,
        description="Most relevant achievements for this job"
    )

    duration_months: Optional[int] = Field(
        default=None,
        ge=0,
        description="Duration in months"
    )

    relevance_reason: Optional[str] = Field(
        default=None,
        description="Explanation of why this experience is relevant"
    )

    @validator('company', 'role')
    def validate_required_strings(cls, value: str) -> str:
        """Validate required string fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty")
        return cleaned

    @validator('matching_skills', 'key_achievements')
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


class RecommendedProject(BaseModel):
    """Model for recommended projects in match context."""

    title: str = Field(
        ...,
        min_length=1,
        description="Project title"
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0-1.0)"
    )

    matching_technologies: List[str] = Field(
        default_factory=list,
        description="Technologies that match job requirements"
    )

    description: Optional[str] = Field(
        default=None,
        description="Brief project description"
    )

    relevance_reason: Optional[str] = Field(
        default=None,
        description="Explanation of why this project is relevant"
    )

    url: Optional[str] = Field(
        default=None,
        description="Project URL if available"
    )

    @validator('title')
    def validate_title(cls, value: str) -> str:
        """Validate project title."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Project title cannot be empty")
        return cleaned

    @validator('matching_technologies')
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


class ImprovementSuggestion(BaseModel):
    """Model for improvement suggestions."""

    category: SuggestionCategory = Field(
        ...,
        description="Category of the suggestion"
    )

    priority: str = Field(
        ...,
        description="Priority level (high, medium, low)"
    )

    suggestion: str = Field(
        ...,
        min_length=10,
        description="Detailed improvement suggestion"
    )

    impact: Optional[str] = Field(
        default=None,
        description="Expected impact of implementing this suggestion"
    )

    resources: List[str] = Field(
        default_factory=list,
        description="Recommended resources or learning materials"
    )

    @validator('priority')
    def validate_priority(cls, value: str) -> str:
        """Validate priority level."""
        valid_priorities = {'high', 'medium', 'low'}
        cleaned = value.lower().strip()
        if cleaned not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return cleaned

    @validator('suggestion')
    def validate_suggestion(cls, value: str) -> str:
        """Validate suggestion text."""
        cleaned = value.strip()
        if len(cleaned) < 10:
            raise ValueError("Suggestion must be at least 10 characters long")
        return cleaned

    @validator('resources')
    def validate_resources(cls, value: List[str]) -> List[str]:
        """Validate resources list."""
        cleaned_items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All resources must be strings")
            cleaned_item = item.strip()
            if cleaned_item:
                cleaned_items.append(cleaned_item)
        return cleaned_items


class AnalysisDetails(BaseModel):
    """Detailed analysis information."""

    skills_match_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of required skills matched"
    )

    experience_match_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of experience requirements matched"
    )

    education_match: bool = Field(
        default=False,
        description="Whether education requirements are met"
    )

    missing_skills: List[str] = Field(
        default_factory=list,
        description="Skills required by job but missing from profile"
    )

    additional_skills: List[str] = Field(
        default_factory=list,
        description="User skills not required but potentially valuable"
    )

    experience_gap_months: Optional[int] = Field(
        default=None,
        ge=0,
        description="Experience gap in months if any"
    )

    strength_areas: List[str] = Field(
        default_factory=list,
        description="Areas where the candidate is particularly strong"
    )

    weakness_areas: List[str] = Field(
        default_factory=list,
        description="Areas that need improvement"
    )

    match_explanation: Optional[str] = Field(
        default=None,
        description="Detailed explanation of the match score"
    )

    @validator('missing_skills', 'additional_skills', 'strength_areas', 'weakness_areas')
    def validate_string_lists(cls, value: List[str]) -> List[str]:
        """Validate string lists."""
        cleaned_items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("All items must be strings")
            cleaned_item = item.strip()
            if cleaned_item:
                cleaned_items.append(cleaned_item)
        return cleaned_items


class MatchResult(BaseModel):
    """
    Comprehensive job-profile match result model.

    This model represents the complete analysis of how well a user profile
    matches a job posting, including quantitative scores, qualitative assessments,
    and actionable recommendations for improvement.

    Attributes:
        score: Overall match score (1-10)
        matched_skills: List of skills that match job requirements
        relevant_experiences: Work experiences relevant to the job
        recommended_projects: Projects that support the application
        suggestions: List of improvement suggestions
        analysis_details: Detailed quantitative analysis
        job_title: Title of the job being matched against
        company_name: Name of the company
        analyzed_at: Timestamp when analysis was performed
    """

    score: MatchScore = Field(
        ...,
        description="Overall match score (1-10)"
    )

    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills that match job requirements"
    )

    relevant_experiences: List[RelevantExperience] = Field(
        default_factory=list,
        description="Work experiences relevant to the job"
    )

    recommended_projects: List[RecommendedProject] = Field(
        default_factory=list,
        description="Projects that support the application"
    )

    suggestions: List[ImprovementSuggestion] = Field(
        default_factory=list,
        description="Actionable improvement suggestions"
    )

    analysis_details: AnalysisDetails = Field(
        ...,
        description="Detailed quantitative analysis"
    )

    job_title: str = Field(
        ...,
        min_length=1,
        description="Title of the job being matched against"
    )

    company_name: str = Field(
        ...,
        min_length=1,
        description="Name of the company"
    )

    analyzed_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when analysis was performed"
    )

    confidence_level: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level of the analysis (0.0-1.0)"
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @validator('job_title', 'company_name')
    def validate_required_strings(cls, value: str) -> str:
        """Validate required string fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty")
        return cleaned

    @validator('matched_skills')
    def validate_matched_skills(cls, value: List[str]) -> List[str]:
        """Validate and clean matched skills list."""
        cleaned_items = []
        seen_skills = set()

        for skill in value:
            if not isinstance(skill, str):
                raise ValueError("All skills must be strings")
            cleaned_skill = skill.strip()
            if cleaned_skill and cleaned_skill.lower() not in seen_skills:
                cleaned_items.append(cleaned_skill)
                seen_skills.add(cleaned_skill.lower())

        return cleaned_items

    @validator('relevant_experiences')
    def validate_experiences(cls, value: List[RelevantExperience]) -> List[RelevantExperience]:
        """Sort experiences by relevance score."""
        return sorted(value, key=lambda x: x.relevance_score, reverse=True)

    @validator('recommended_projects')
    def validate_projects(cls, value: List[RecommendedProject]) -> List[RecommendedProject]:
        """Sort projects by relevance score."""
        return sorted(value, key=lambda x: x.relevance_score, reverse=True)

    @validator('suggestions')
    def validate_suggestions(cls, value: List[ImprovementSuggestion]) -> List[ImprovementSuggestion]:
        """Sort suggestions by priority."""
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        return sorted(value, key=lambda x: priority_order.get(x.priority, 0), reverse=True)

    @validator('analyzed_at')
    def validate_analyzed_at(cls, value: datetime) -> datetime:
        """Ensure analysis timestamp is not in the future."""
        if value > datetime.now():
            raise ValueError("Analysis timestamp cannot be in the future")
        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary with proper serialization.

        Returns:
            Dictionary representation of the match result
        """
        data = self.dict()

        # Convert datetime to ISO string
        if isinstance(data.get('analyzed_at'), datetime):
            data['analyzed_at'] = data['analyzed_at'].isoformat()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchResult':
        """
        Create a MatchResult instance from a dictionary.

        Args:
            data: Dictionary containing match result data

        Returns:
            MatchResult instance

        Raises:
            ValueError: If data is invalid or missing required fields
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        # Handle datetime conversion
        if 'analyzed_at' in data and isinstance(data['analyzed_at'], str):
            try:
                data['analyzed_at'] = datetime.fromisoformat(data['analyzed_at'])
            except ValueError as e:
                raise ValueError(f"Invalid datetime format for analyzed_at: {e}")

        return cls(**data)

    def is_good_match(self, threshold: int = 6) -> bool:
        """
        Check if the match score meets or exceeds the threshold.

        Args:
            threshold: Minimum score to consider a good match (default: 6)

        Returns:
            True if score is at or above threshold
        """
        return self.score >= threshold

    def get_match_category(self) -> str:
        """
        Get a descriptive category for the match score.

        Returns:
            String describing the match quality
        """
        if self.score >= 9:
            return "Excellent Match"
        elif self.score >= 7:
            return "Good Match"
        elif self.score >= 5:
            return "Average Match"
        elif self.score >= 3:
            return "Poor Match"
        else:
            return "Very Poor Match"

    def get_top_suggestions(self, limit: int = 3) -> List[ImprovementSuggestion]:
        """
        Get the top priority suggestions.

        Args:
            limit: Maximum number of suggestions to return

        Returns:
            List of top priority suggestions
        """
        return self.suggestions[:limit]

    def get_skills_gap(self) -> Dict[str, Any]:
        """
        Get detailed skills gap analysis.

        Returns:
            Dictionary with skills gap information
        """
        return {
            'matched_skills': self.matched_skills,
            'missing_skills': self.analysis_details.missing_skills,
            'additional_skills': self.analysis_details.additional_skills,
            'match_percentage': self.analysis_details.skills_match_percentage
        }

    def get_summary(self) -> str:
        """
        Generate a concise summary of the match result.

        Returns:
            Formatted summary string
        """
        category = self.get_match_category()
        skills_matched = len(self.matched_skills)
        experiences_count = len(self.relevant_experiences)

        return (
            f"{category} ({self.score}/10) for {self.job_title} at {self.company_name}. "
            f"Matched {skills_matched} skills, {experiences_count} relevant experiences. "
            f"Skills match: {self.analysis_details.skills_match_percentage:.1f}%"
        )

    def get_recommendations_by_category(self) -> Dict[str, List[ImprovementSuggestion]]:
        """
        Group recommendations by category.

        Returns:
            Dictionary with suggestions grouped by category
        """
        grouped = {}
        for suggestion in self.suggestions:
            category = suggestion.category.value
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(suggestion)

        return grouped

    def __str__(self) -> str:
        """String representation of the match result."""
        return f"MatchResult(score={self.score}, job='{self.job_title}', company='{self.company_name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the match result."""
        return (
            f"MatchResult(score={self.score}, "
            f"matched_skills={len(self.matched_skills)}, "
            f"relevant_experiences={len(self.relevant_experiences)}, "
            f"suggestions={len(self.suggestions)}, "
            f"confidence={self.confidence_level:.2f})"
        )