"""
CV Generator Agents Package

This package contains all the intelligent agents for the CV generation workflow:
- JobAnalyzer: Extracts and structures job posting data
- ProfileMatcher: Analyzes job-profile compatibility
- TemplateCustomizer: Personalizes CV/cover letter templates
- PreviewGenerator: Creates interactive previews
- PDFGenerator: Generates final PDF documents

Each agent is designed to work independently while integrating seamlessly
with the LangGraph workflow orchestration system.
"""

# Make agents available at package level
try:
    from .job_analyzer import JobAnalyzer
    from .profile_matcher import ProfileMatcher

    __all__ = ['JobAnalyzer', 'ProfileMatcher']
except ImportError:
    # Handle import errors gracefully during development
    pass