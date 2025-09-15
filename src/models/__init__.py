"""
Models package for the CV/Cover Letter generator system.

This package contains comprehensive Pydantic models for:
- JobData: Job posting information with validation
- UserProfile: Complete user profile with nested models
- MatchResult: Job-profile matching analysis results

All models include proper validation, serialization, and comprehensive documentation.
"""

from .job_data import JobData, ExperienceLevel
from .user_profile import (
    UserProfile,
    PersonalInfo,
    WorkExperience,
    Education,
    Project,
    SocialUrls
)
from .match_result import (
    MatchResult,
    MatchScore,
    RelevantExperience,
    RecommendedProject,
    ImprovementSuggestion,
    AnalysisDetails,
    SuggestionCategory
)

__all__ = [
    # Job Data Models
    "JobData",
    "ExperienceLevel",

    # User Profile Models
    "UserProfile",
    "PersonalInfo",
    "WorkExperience",
    "Education",
    "Project",
    "SocialUrls",

    # Match Result Models
    "MatchResult",
    "MatchScore",
    "RelevantExperience",
    "RecommendedProject",
    "ImprovementSuggestion",
    "AnalysisDetails",
    "SuggestionCategory"
]

# Version information
__version__ = "1.0.0"