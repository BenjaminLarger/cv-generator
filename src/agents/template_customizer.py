"""
Single-Page PDF Optimized Template Customization System for CV/Cover Letter Generation.

This module provides intelligent template customization with strict single-page PDF constraints.
It prioritizes content, enforces character limits, and generates compelling documents that
showcase the most relevant qualifications within tight space constraints for PDF conversion.

Key Features:
- Strict single-page PDF constraints with character limits
- Content prioritization based on job matching analysis
- Intelligent content truncation preserving meaning
- PDF-optimized template design with compact spacing
- Content length validation and space utilization checks
"""

import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.match_result import MatchResult
from models.user_profile import UserProfile
from models.job_data import JobData
from utils.template_engine import load_template, replace_placeholders, TemplateError
from utils.logging_config import get_agents_logger

logger = get_agents_logger()


# CRITICAL: Content limits for single-page PDF optimization
CONTENT_LIMITS = {
    'professional_summary': 300,           # Professional summary max characters
    'project_description': 120,            # Project description max characters
    'achievement_bullet': 80,              # Achievement bullet point max characters
    'cover_letter_opening': 200,           # Cover letter opening paragraph max characters
    'cover_letter_body_paragraph': 200,    # Cover letter body paragraph max characters
    'cover_letter_closing': 150,           # Cover letter closing paragraph max characters
    'skills_list': 12,                     # Maximum number of skills to display
    'experiences_shown': 3,                # Maximum work experiences to show
    'projects_shown': 3,                   # Maximum projects to show
    'achievements_per_experience': 2,      # Maximum achievements per experience
    'skills_per_category': 8,              # Maximum skills per category
    'total_cv_sections': 6,                # Maximum CV sections for space management
    'cover_letter_achievements': 3,        # Maximum achievements in cover letter
    'experience_title_length': 60,         # Maximum length for experience titles
    'project_title_length': 40,            # Maximum length for project titles
}


class SinglePageConstraints(BaseModel):
    """Model for single-page PDF constraints and validation."""

    max_experiences: int = Field(default=3, ge=1, le=5, description="Maximum experiences to show")
    max_projects: int = Field(default=3, ge=1, le=4, description="Maximum projects to show")
    max_skills_per_category: int = Field(default=8, ge=5, le=12, description="Maximum skills per category")
    professional_summary_limit: int = Field(default=300, ge=200, le=400, description="Professional summary character limit")
    project_description_limit: int = Field(default=120, ge=80, le=160, description="Project description character limit")
    achievement_limit: int = Field(default=80, ge=50, le=100, description="Achievement bullet character limit")

    def validate_content_fits_page(self, content_metrics: Dict[str, int]) -> bool:
        """Validate that content will fit on a single page."""
        # Estimated character limits for single-page PDF
        total_chars = sum(content_metrics.values())
        return total_chars <= 3500  # Conservative estimate for single page


class ContentPriority(BaseModel):
    """Model for prioritized content selection."""

    experiences: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prioritized work experiences"
    )

    projects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prioritized projects (max 3)"
    )

    skills: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Categorized and prioritized skills"
    )

    achievements: List[str] = Field(
        default_factory=list,
        description="Most relevant achievements"
    )


class CustomizationResult(BaseModel):
    """Result of template customization process."""

    cv_html: str = Field(..., description="Customized CV HTML")
    cover_letter_html: str = Field(..., description="Customized cover letter HTML")
    changes_made: List[str] = Field(default_factory=list, description="List of changes applied")
    customization_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score indicating level of customization"
    )
    single_page_compliant: bool = Field(default=False, description="Whether content fits single page constraints")
    content_metrics: Dict[str, int] = Field(default_factory=dict, description="Content length metrics")
    space_utilization: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated page space utilization")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True


class TemplateCustomizer:
    """
    Intelligent template customization system for CV and cover letters.

    This class provides sophisticated template customization based on job matching
    analysis, prioritizing relevant content and generating personalized documents.
    """

    def __init__(self, templates_dir: Optional[str] = None, single_page_mode: bool = True):
        """
        Initialize the template customizer.

        Args:
            templates_dir: Directory containing template files
            single_page_mode: Enable strict single-page PDF constraints
        """
        if templates_dir is None:
            # Default to templates directory relative to this file
            current_dir = Path(__file__).parent.parent.parent
            templates_dir = current_dir / "templates"

        self.templates_dir = Path(templates_dir)
        self.single_page_mode = single_page_mode
        self.constraints = SinglePageConstraints() if single_page_mode else None
        logger.info(f"Initialized TemplateCustomizer with templates dir: {self.templates_dir}, single_page_mode: {single_page_mode}")

    def customize_templates(
        self,
        match_result: MatchResult,
        user_profile: UserProfile,
        job_data: JobData
    ) -> CustomizationResult:
        """
        Main entry point that customizes both CV and cover letter templates.

        Args:
            match_result: Job matching analysis results
            user_profile: User's profile data
            job_data: Target job information

        Returns:
            CustomizationResult with customized HTML and compliance metrics

        Raises:
            TemplateError: If template customization fails
        """
        logger.info(f"Starting template customization for {job_data.position} at {job_data.company_name}")

        try:
            # Generate dynamic content based on match results
            dynamic_content = self.generate_dynamic_content(match_result, user_profile, job_data)

            # Load templates
            cv_template_path = self.templates_dir / "cv_template.html"
            cover_letter_template_path = self.templates_dir / "cover_letter_template.html"

            cv_template = load_template(str(cv_template_path))
            cover_letter_template = load_template(str(cover_letter_template_path))

            # Optimize content for single-page constraints if enabled
            if self.single_page_mode:
                dynamic_content = self._optimize_for_single_page(dynamic_content)

            # Customize templates
            cv_html = self.customize_cv(cv_template, dynamic_content)
            cover_letter_html = self.customize_cover_letter(cover_letter_template, dynamic_content)

            # Track changes and validate constraints
            changes = self.track_changes(cv_template + cover_letter_template, cv_html + cover_letter_html)
            content_metrics = self._calculate_content_metrics(dynamic_content)
            space_utilization = self._estimate_space_utilization(content_metrics)
            single_page_compliant = self._validate_single_page_compliance(content_metrics)

            customization_score = self._calculate_customization_score(match_result, dynamic_content)

            logger.info(f"Template customization completed - Single page compliant: {single_page_compliant}, Space utilization: {space_utilization:.2f}")

            return CustomizationResult(
                cv_html=cv_html,
                cover_letter_html=cover_letter_html,
                changes_made=changes,
                customization_score=customization_score,
                single_page_compliant=single_page_compliant,
                content_metrics=content_metrics,
                space_utilization=space_utilization
            )

        except Exception as e:
            error_msg = f"Failed to customize templates: {str(e)}"
            logger.error(error_msg)
            raise TemplateError(error_msg)

    def customize_cv(self, template: str, data: Dict[str, Any]) -> str:
        """
        Customize CV template with prioritized content based on job matching.

        CRITICAL: Enforces single-page constraints with character limits.

        Args:
            template: CV template HTML string
            data: Template data generated from match analysis

        Returns:
            Customized CV HTML string optimized for single-page PDF
        """
        logger.info("Customizing CV template with single-page constraints")

        try:
            # Apply single-page optimizations to data
            if self.single_page_mode:
                data = self._apply_cv_constraints(data)

            # Replace placeholders with prioritized content
            customized_cv = replace_placeholders(template, data, strict_mode=False)

            # Validate single-page compliance
            if self.single_page_mode:
                content_length = len(customized_cv)
                logger.info(f"CV content length: {content_length} characters")
                if content_length > 6000:  # Conservative estimate for CV content
                    logger.warning(f"CV content may exceed single page: {content_length} characters")

            logger.info("CV template customization completed")
            return customized_cv

        except Exception as e:
            error_msg = f"Failed to customize CV template: {str(e)}"
            logger.error(error_msg)
            raise TemplateError(error_msg)

    def customize_cover_letter(self, template: str, data: Dict[str, Any]) -> str:
        """
        Generate personalized cover letter content.

        CRITICAL: Maintains single-page format with optimized content length.

        Args:
            template: Cover letter template HTML string
            data: Template data with personalized content

        Returns:
            Customized cover letter HTML string optimized for single-page PDF
        """
        logger.info("Customizing cover letter template with single-page constraints")

        try:
            # Apply single-page optimizations to data
            if self.single_page_mode:
                data = self._apply_cover_letter_constraints(data)

            # Replace placeholders with personalized content
            customized_letter = replace_placeholders(template, data, strict_mode=False)

            # Validate single-page compliance
            if self.single_page_mode:
                content_length = len(customized_letter)
                logger.info(f"Cover letter content length: {content_length} characters")
                if content_length > 4000:  # Conservative estimate for cover letter
                    logger.warning(f"Cover letter content may exceed single page: {content_length} characters")

            logger.info("Cover letter template customization completed")
            return customized_letter

        except Exception as e:
            error_msg = f"Failed to customize cover letter template: {str(e)}"
            logger.error(error_msg)
            raise TemplateError(error_msg)

    def generate_dynamic_content(
        self,
        match_result: MatchResult,
        user_profile: UserProfile,
        job_data: JobData
    ) -> Dict[str, Any]:
        """
        Transform MatchResult into template-ready content.

        Args:
            match_result: Job matching analysis results
            user_profile: User's profile data
            job_data: Target job information

        Returns:
            Dictionary of template-ready content with prioritized data
        """
        logger.info("Generating dynamic content from match analysis")

        # Extract basic personal information
        personal = user_profile.personal_info

        # Prioritize content based on match results with single-page constraints
        prioritized_experiences = self._prioritize_experiences(
            match_result.relevant_experiences,
            user_profile.experiences,
            max_count=CONTENT_LIMITS['experiences_shown']
        )

        prioritized_projects = self._prioritize_projects(
            match_result.recommended_projects,
            user_profile.projects,
            max_count=CONTENT_LIMITS['projects_shown']
        )

        prioritized_skills = self._prioritize_skills(
            match_result.matched_skills,
            user_profile.skills,
            job_data.skills_required,
            max_skills=CONTENT_LIMITS['skills_list']
        )

        # Generate cover letter specific content
        cover_letter_content = self._generate_cover_letter_content(
            match_result, user_profile, job_data
        )

        # Create comprehensive data dictionary
        data = {
            # Personal Information
            'FULL_NAME': personal.name,
            'EMAIL': str(personal.email),
            'PHONE': personal.phone or '',
            'LOCATION': personal.location or '',
            'LINKEDIN_URL': str(user_profile.urls.linkedin) if user_profile.urls.linkedin else '',
            'GITHUB_URL': str(user_profile.urls.github) if user_profile.urls.github else '',

            # Job-specific information
            'JOB_TITLE': job_data.position,
            'COMPANY_NAME': job_data.company_name,

            # CV Content - Length optimized for single page
            'PROFESSIONAL_SUMMARY': self._truncate_with_ellipsis(
                self._generate_professional_summary(match_result, user_profile, job_data),
                CONTENT_LIMITS['professional_summary']
            ),
            'RELEVANT_EXPERIENCES': self._format_experiences_html(prioritized_experiences),
            'SELECTED_PROJECTS': self._format_projects_html(prioritized_projects),
            'RELEVANT_SKILLS': ', '.join(prioritized_skills['technical'][:CONTENT_LIMITS['skills_per_category']]),
            'TECHNICAL_SKILLS': ', '.join(prioritized_skills['technical'][:3]),  # Top 3 for existing template
            'RELEVANT_TOOLS': ', '.join(prioritized_skills['tools'][:5]),  # Top 5 tools

            # Dynamic project placeholders for existing template - Length constrained
            'TITLE_OF_THE_SIDE_PROJECT_1': self._truncate_with_ellipsis(
                prioritized_projects[0]['title'] if prioritized_projects else '',
                CONTENT_LIMITS['project_title_length']
            ),
            'DESCRIPTION_OF_THE_SIDE_PROJECT_1': self._truncate_with_ellipsis(
                prioritized_projects[0]['description'] if prioritized_projects else '',
                CONTENT_LIMITS['project_description']
            ),
            'TITLE_OF_THE_SIDE_PROJECT_2': self._truncate_with_ellipsis(
                prioritized_projects[1]['title'] if len(prioritized_projects) > 1 else '',
                CONTENT_LIMITS['project_title_length']
            ),
            'DESCRIPTION_OF_THE_SIDE_PROJECT_2': self._truncate_with_ellipsis(
                prioritized_projects[1]['description'] if len(prioritized_projects) > 1 else '',
                CONTENT_LIMITS['project_description']
            ),

            # Cover Letter Content - Length optimized for single page
            'INSERT_DATE': datetime.now().strftime("%B %d, %Y"),
            'INSERT_COMPANY_NAME': job_data.company_name,
            'INSERT_JOB_TITLE': job_data.position,
            'OPENING_PARAGRAPH': self._truncate_with_ellipsis(
                cover_letter_content['opening'],
                CONTENT_LIMITS['cover_letter_opening']
            ),
            'EXPERIENCE_MATCH_PARAGRAPH': self._truncate_with_ellipsis(
                cover_letter_content['experience_match'],
                CONTENT_LIMITS['cover_letter_body_paragraph']
            ),
            'SKILLS_PARAGRAPH': self._truncate_with_ellipsis(
                cover_letter_content['skills_paragraph'],
                CONTENT_LIMITS['cover_letter_body_paragraph']
            ),
            'CLOSING_PARAGRAPH': self._truncate_with_ellipsis(
                cover_letter_content['closing'],
                CONTENT_LIMITS['cover_letter_closing']
            ),
            'INSERT_ACHIEVEMENT_1': cover_letter_content['achievements'][0] if len(cover_letter_content['achievements']) > 0 else '',
            'INSERT_ACHIEVEMENT_2': cover_letter_content['achievements'][1] if len(cover_letter_content['achievements']) > 1 else '',
            'INSERT_ACHIEVEMENT_3': cover_letter_content['achievements'][2] if len(cover_letter_content['achievements']) > 2 else '',
            'INSERT_RELEVANT_SKILLS': ', '.join(prioritized_skills['technical'][:5]),
            'INSERT_SPECIFIC_DETAIL_ABOUT_COMPANY': cover_letter_content['company_interest'],
            'INSERT_REASON_DRAWN_TO_ROLE': cover_letter_content['role_interest'],
            'INSERT_SPECIFIC_PROJECT_OR_GOAL': cover_letter_content['contribution_target'],
        }

        logger.info(f"Generated dynamic content with {len(data)} template variables")
        return data

    def _prioritize_experiences(
        self,
        relevant_experiences: List[Any],
        all_experiences: List[Any],
        max_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Prioritize work experiences based on relevance scores."""
        logger.debug("Prioritizing work experiences")

        # Convert relevant experiences to a lookup dict
        relevant_lookup = {}
        for exp in relevant_experiences:
            key = f"{exp.company.lower()}_{exp.role.lower()}"
            relevant_lookup[key] = exp

        prioritized = []

        # Add relevant experiences first (sorted by relevance score)
        for exp in sorted(relevant_experiences, key=lambda x: x.relevance_score, reverse=True):
            duration_months = exp.duration_months or 12  # Default if not specified

            prioritized.append({
                'company': exp.company,
                'role': exp.role,
                'relevance_score': exp.relevance_score,
                'matching_skills': exp.matching_skills,
                'key_achievements': exp.key_achievements,
                'duration_months': duration_months,
                'relevance_reason': exp.relevance_reason
            })

        # Add other experiences (up to max_count total)
        if len(prioritized) < max_count:
            for exp in all_experiences:
                key = f"{exp.company.lower()}_{exp.role.lower()}"
                if key not in relevant_lookup and len(prioritized) < max_count:
                    achievements = [self._truncate_with_ellipsis(str(achievement), CONTENT_LIMITS['achievement_bullet'])
                                  for achievement in exp.achievements[:CONTENT_LIMITS['achievements_per_experience']]]

                    prioritized.append({
                        'company': exp.company,
                        'role': exp.role,
                        'relevance_score': 0.1,  # Low relevance for non-matched experiences
                        'matching_skills': [],
                        'key_achievements': achievements,
                        'duration_months': exp.get_duration_months() or 12,
                        'relevance_reason': 'Additional experience'
                    })

        logger.debug(f"Prioritized {len(prioritized)} work experiences")
        return prioritized[:max_count]  # Limit to max_count

    def _prioritize_projects(
        self,
        recommended_projects: List[Any],
        all_projects: List[Any],
        max_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Prioritize projects based on relevance scores."""
        logger.debug("Prioritizing projects")

        # Convert recommended projects to a lookup dict
        recommended_lookup = {proj.title.lower(): proj for proj in recommended_projects}

        prioritized = []

        # Add recommended projects first (sorted by relevance score)
        for project in sorted(recommended_projects, key=lambda x: x.relevance_score, reverse=True):
            description = self._truncate_description(
                project.description or "Relevant project showcasing required skills",
                80, CONTENT_LIMITS['project_description']
            )

            prioritized.append({
                'title': project.title,
                'description': description,
                'relevance_score': project.relevance_score,
                'matching_technologies': project.matching_technologies,
                'url': str(project.url) if project.url else None,
                'relevance_reason': project.relevance_reason
            })

        # Add other projects if needed (up to max_count total)
        if len(prioritized) < max_count:
            for project in all_projects:
                if project.title.lower() not in recommended_lookup and len(prioritized) < max_count:
                    description = self._truncate_description(project.description, 80, CONTENT_LIMITS['project_description'])

                    prioritized.append({
                        'title': project.title,
                        'description': description,
                        'relevance_score': 0.1,  # Low relevance for non-recommended projects
                        'matching_technologies': project.technologies[:3],  # Use some technologies
                        'url': str(project.url) if project.url else None,
                        'relevance_reason': 'Additional project experience'
                    })

        logger.debug(f"Prioritized {len(prioritized)} projects")
        return prioritized[:max_count]  # Limit to max_count

    def _prioritize_skills(
        self,
        matched_skills: List[str],
        user_skills: List[str],
        job_skills: List[str],
        max_skills: int = 12
    ) -> Dict[str, List[str]]:
        """Categorize and prioritize skills based on job matching."""
        logger.debug("Prioritizing and categorizing skills")

        # Define skill categories
        technical_keywords = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'git', 'jenkins', 'gitlab', 'github', 'ci/cd', 'devops',
            'django', 'flask', 'fastapi', 'spring', 'express',
            'html', 'css', 'scss', 'bootstrap', 'tailwind',
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
            'machine learning', 'data science', 'ai', 'ml'
        ]

        tool_keywords = [
            'jira', 'confluence', 'slack', 'trello', 'notion',
            'postman', 'swagger', 'figma', 'sketch',
            'vscode', 'intellij', 'pycharm', 'jupyter',
            'tableau', 'power bi', 'excel', 'sheets'
        ]

        # Categorize skills
        technical_skills = []
        tools = []
        soft_skills = []

        # Prioritize matched skills first
        all_skills_to_process = []

        # Add matched skills with high priority
        for skill in matched_skills:
            if skill in user_skills:
                all_skills_to_process.append((skill, 2))  # High priority

        # Add job required skills that user has
        for skill in job_skills:
            if skill in user_skills and skill not in matched_skills:
                all_skills_to_process.append((skill, 1))  # Medium priority

        # Add remaining user skills
        for skill in user_skills:
            if skill not in [s[0] for s in all_skills_to_process]:
                all_skills_to_process.append((skill, 0))  # Low priority

        # Sort by priority and categorize
        all_skills_to_process.sort(key=lambda x: x[1], reverse=True)

        for skill, priority in all_skills_to_process:
            skill_lower = skill.lower()

            # Check if it's a technical skill
            if any(tech_keyword in skill_lower for tech_keyword in technical_keywords):
                technical_skills.append(skill)
            # Check if it's a tool
            elif any(tool_keyword in skill_lower for tool_keyword in tool_keywords):
                tools.append(skill)
            else:
                soft_skills.append(skill)

        result = {
            'technical': technical_skills[:max_skills],
            'tools': tools[:max_skills//2],
            'soft': soft_skills[:max_skills//3]
        }

        logger.debug(f"Categorized skills: {len(technical_skills)} technical, {len(tools)} tools, {len(soft_skills)} soft")
        return result

    def _generate_cover_letter_content(
        self,
        match_result: MatchResult,
        user_profile: UserProfile,
        job_data: JobData
    ) -> Dict[str, Any]:
        """Generate personalized cover letter content paragraphs."""
        logger.debug("Generating cover letter content")

        # Extract top achievements from relevant experiences
        achievements = []
        for exp in match_result.relevant_experiences[:3]:  # Top 3 relevant experiences
            achievements.extend(exp.key_achievements[:2])  # Up to 2 achievements each

        # If we don't have enough achievements, add from analysis details
        if len(achievements) < 3:
            achievements.extend(match_result.analysis_details.strength_areas[:3])

        # Generate personalized paragraphs
        content = {
            'opening': self._generate_opening_paragraph(job_data, user_profile),
            'experience_match': self._generate_experience_match_paragraph(match_result, job_data),
            'skills_paragraph': self._generate_skills_paragraph(match_result, job_data),
            'closing': self._generate_closing_paragraph(job_data),
            'achievements': achievements[:3],  # Top 3 achievements
            'company_interest': self._generate_company_interest(job_data),
            'role_interest': self._generate_role_interest(job_data, match_result),
            'contribution_target': self._generate_contribution_target(job_data)
        }

        logger.debug("Generated cover letter content sections")
        return content

    def _generate_professional_summary(
        self,
        match_result: MatchResult,
        user_profile: UserProfile,
        job_data: JobData
    ) -> str:
        """Generate a job-specific professional summary."""
        experience_years = user_profile.get_total_experience_years()
        top_skills = match_result.matched_skills[:3]  # Top 3 matched skills

        # Create a compelling summary based on match results
        if match_result.score >= 8:
            summary = f"Experienced {job_data.position.lower()} with {experience_years} years of expertise in {', '.join(top_skills)}. "
            summary += f"Proven track record in {match_result.analysis_details.strength_areas[0] if match_result.analysis_details.strength_areas else 'technology solutions'}, "
            summary += f"seeking to leverage deep technical knowledge and leadership skills at {job_data.company_name}."
        elif match_result.score >= 6:
            summary = f"Tech-savvy professional with {experience_years} years of experience and strong skills in {', '.join(top_skills)}. "
            summary += f"Looking to apply proven abilities in {match_result.analysis_details.strength_areas[0] if match_result.analysis_details.strength_areas else 'software development'} "
            summary += f"as {job_data.position} at {job_data.company_name}."
        else:
            summary = f"Motivated technology professional with {experience_years} years of experience in {', '.join(top_skills[:2] if len(top_skills) >= 2 else user_profile.skills[:2])}. "
            summary += f"Eager to transition skills and drive impact as {job_data.position} in a dynamic, innovative environment at {job_data.company_name}."

        return summary

    def _generate_opening_paragraph(self, job_data: JobData, user_profile: UserProfile) -> str:
        """Generate personalized opening paragraph for cover letter."""
        return (
            f"I am writing to express my strong interest in the {job_data.position} position at {job_data.company_name}. "
            f"With my background in programming engineering from 42 School and {user_profile.get_total_experience_years()} years of "
            f"hands-on experience in technology solutions, I am excited about the opportunity to contribute to your innovative team."
        )

    def _generate_experience_match_paragraph(self, match_result: MatchResult, job_data: JobData) -> str:
        """Generate paragraph highlighting relevant experience matches."""
        if not match_result.relevant_experiences:
            return (
                f"My diverse technical background and strong foundation in software development "
                f"position me well for the challenges of the {job_data.position} role."
            )

        top_experience = match_result.relevant_experiences[0]

        return (
            f"My experience as {top_experience.role} at {top_experience.company} directly aligns with your requirements. "
            f"In this role, I {top_experience.key_achievements[0] if top_experience.key_achievements else 'developed innovative solutions'}, "
            f"demonstrating expertise in {', '.join(top_experience.matching_skills[:3])} that would be valuable for this position."
        )

    def _generate_skills_paragraph(self, match_result: MatchResult, job_data: JobData) -> str:
        """Generate paragraph showcasing matching skills."""
        matched_skills = match_result.matched_skills[:5]  # Top 5 matched skills
        match_percentage = match_result.analysis_details.skills_match_percentage

        return (
            f"I bring strong technical capabilities in {', '.join(matched_skills)}, achieving a {match_percentage:.0f}% "
            f"match with your technical requirements. My proven ability to {match_result.analysis_details.strength_areas[0].lower() if match_result.analysis_details.strength_areas else 'deliver results'} "
            f"and adapt to new technologies makes me well-suited for the dynamic challenges at {job_data.company_name}."
        )

    def _generate_closing_paragraph(self, job_data: JobData) -> str:
        """Generate compelling closing paragraph."""
        return (
            f"I am excited about the opportunity to bring my technical expertise and passion for innovation to {job_data.company_name}. "
            f"I would welcome the chance to discuss how my background and enthusiasm can contribute to your team's continued success."
        )

    def _generate_company_interest(self, job_data: JobData) -> str:
        """Generate specific interest in the company."""
        return f"the innovative work {job_data.company_name} is doing in {job_data.experience_level} technology solutions"

    def _generate_role_interest(self, job_data: JobData, match_result: MatchResult) -> str:
        """Generate specific reason for interest in the role."""
        if match_result.analysis_details.strength_areas:
            return f"it perfectly aligns with my strengths in {match_result.analysis_details.strength_areas[0].lower()} and my career goals"
        return f"it offers the perfect opportunity to apply my technical skills in a challenging environment"

    def _generate_contribution_target(self, job_data: JobData) -> str:
        """Generate specific contribution target."""
        return f"your technology initiatives and {job_data.experience_level}-level development projects"

    def _format_experiences_html(self, experiences: List[Dict[str, Any]]) -> str:
        """Format work experiences as HTML for template insertion."""
        if not experiences:
            return "<p>No relevant experience data available.</p>"

        html_parts = []
        for exp in experiences:
            achievements_html = ""
            if exp['key_achievements']:
                achievements_list = '\n'.join([f"        <li>{achievement}</li>" for achievement in exp['key_achievements']])
                achievements_html = f"    <ul>\n{achievements_list}\n    </ul>"

            exp_html = f"""
    <div class="experience-item">
        <div class="item-title">{exp['role']} at {exp['company']}</div>
        <div class="item-subtitle">Relevance Score: {exp['relevance_score']:.1f}</div>
{achievements_html}
        <p><strong>Key Skills:</strong> {', '.join(exp['matching_skills'][:5])}</p>
    </div>"""
            html_parts.append(exp_html)

        return '\n'.join(html_parts)

    def _format_projects_html(self, projects: List[Dict[str, Any]]) -> str:
        """Format projects as HTML for template insertion."""
        if not projects:
            return "<p>No relevant projects data available.</p>"

        html_parts = []
        for project in projects:
            project_html = f"""
    <div class="project-item">
        <div class="item-title">{project['title']}</div>
        <p>{project['description']}</p>
        <p><strong>Technologies:</strong> {', '.join(project['matching_technologies'][:5])}</p>
        {f'<p><strong>URL:</strong> <a href="{project["url"]}" target="_blank">{project["url"]}</a></p>' if project['url'] else ''}
    </div>"""
            html_parts.append(project_html)

        return '\n'.join(html_parts)

    def _truncate_description(self, description: str, min_length: int = 100, max_length: int = 165) -> str:
        """
        Truncate description to fit within character limits while preserving meaning.

        Args:
            description: Original description
            min_length: Minimum character count
            max_length: Maximum character count

        Returns:
            Truncated description within specified bounds
        """
        if not description:
            return "Innovative project showcasing technical expertise and problem-solving skills in modern technology stack."

        # If description is already within bounds, return as-is
        if min_length <= len(description) <= max_length:
            return description

        # If too short, pad with generic content
        if len(description) < min_length:
            padding = " Demonstrates strong technical skills and attention to detail in implementation."
            while len(description) < min_length and len(description + padding) <= max_length:
                description += padding
                if len(description) >= min_length:
                    break

            # Trim to max if needed
            if len(description) > max_length:
                description = description[:max_length-3] + "..."

        # If too long, truncate intelligently
        elif len(description) > max_length:
            # Try to truncate at sentence boundary
            sentences = description.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + ". ") <= max_length - 3:
                    truncated += sentence + ". "
                else:
                    break

            if truncated and len(truncated) >= min_length:
                description = truncated.rstrip() + "..."
            else:
                # Truncate at word boundary
                words = description.split()
                truncated = ""
                for word in words:
                    if len(truncated + word + " ") <= max_length - 3:
                        truncated += word + " "
                    else:
                        break
                description = truncated.rstrip() + "..."

        return description

    def track_changes(self, original: str, customized: str) -> List[str]:
        """
        Identify and track what content was customized/added.

        Args:
            original: Original template content
            customized: Customized template content

        Returns:
            List of changes for preview display
        """
        logger.debug("Tracking template changes")

        changes = []

        # Find placeholder replacements
        placeholder_pattern = re.compile(r'<!--\s*([A-Z_][A-Z0-9_]*)\s*-->')
        original_placeholders = set(placeholder_pattern.findall(original))
        remaining_placeholders = set(placeholder_pattern.findall(customized))

        replaced_placeholders = original_placeholders - remaining_placeholders

        for placeholder in replaced_placeholders:
            changes.append(f"Replaced placeholder: {placeholder}")

        # Check for content additions
        original_length = len(original)
        customized_length = len(customized)

        if customized_length > original_length:
            changes.append(f"Added {customized_length - original_length} characters of personalized content")

        # Check for specific content types
        if "relevance_score" in customized.lower():
            changes.append("Added relevance scoring for experiences and projects")

        if any(skill_word in customized.lower() for skill_word in ["python", "javascript", "java", "react"]):
            changes.append("Prioritized technical skills based on job requirements")

        logger.debug(f"Tracked {len(changes)} template changes")
        return changes

    def _truncate_with_ellipsis(self, text: str, max_length: int) -> str:
        """
        Truncate text with ellipsis if it exceeds max_length.

        Args:
            text: Text to truncate
            max_length: Maximum character count

        Returns:
            Truncated text with ellipsis if needed
        """
        if not text or max_length <= 0:
            return ""

        if len(text) <= max_length:
            return text

        # Truncate at word boundary if possible
        if max_length > 3:
            truncate_at = max_length - 3
            # Find last space before truncation point
            last_space = text.rfind(' ', 0, truncate_at)
            if last_space > max_length // 2:  # Only use word boundary if it's not too early
                return text[:last_space] + "..."

        return text[:max_length-3] + "..."

    def _prioritize_content_by_relevance(self, items: List[Any], limit: int) -> List[Any]:
        """
        Prioritize content items by relevance score and limit count.

        Args:
            items: List of items with relevance_score attribute
            limit: Maximum number of items to return

        Returns:
            Prioritized and limited list of items
        """
        if not items:
            return []

        # Sort by relevance score if available
        try:
            sorted_items = sorted(items, key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
        except (AttributeError, TypeError):
            sorted_items = items

        return sorted_items[:limit]

    def _optimize_for_single_page(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize content dictionary for single-page constraints.

        Args:
            content: Content dictionary to optimize

        Returns:
            Optimized content dictionary
        """
        if not self.single_page_mode:
            return content

        logger.debug("Optimizing content for single-page constraints")
        optimized = content.copy()

        # Truncate all text fields according to limits
        text_fields = {
            'PROFESSIONAL_SUMMARY': CONTENT_LIMITS['professional_summary'],
            'OPENING_PARAGRAPH': CONTENT_LIMITS['cover_letter_opening'],
            'EXPERIENCE_MATCH_PARAGRAPH': CONTENT_LIMITS['cover_letter_body_paragraph'],
            'SKILLS_PARAGRAPH': CONTENT_LIMITS['cover_letter_body_paragraph'],
            'CLOSING_PARAGRAPH': CONTENT_LIMITS['cover_letter_closing'],
            'DESCRIPTION_OF_THE_SIDE_PROJECT_1': CONTENT_LIMITS['project_description'],
            'DESCRIPTION_OF_THE_SIDE_PROJECT_2': CONTENT_LIMITS['project_description'],
        }

        for field, limit in text_fields.items():
            if field in optimized and isinstance(optimized[field], str):
                optimized[field] = self._truncate_with_ellipsis(optimized[field], limit)

        logger.debug("Content optimization completed")
        return optimized

    def _apply_cv_constraints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CV-specific single-page constraints to data.

        Args:
            data: CV template data

        Returns:
            Constrained data optimized for CV single page
        """
        constrained = data.copy()

        # Limit technical skills display
        if 'TECHNICAL_SKILLS' in constrained:
            skills = constrained['TECHNICAL_SKILLS']
            if isinstance(skills, str) and len(skills) > 100:
                skill_list = [s.strip() for s in skills.split(',')]
                constrained['TECHNICAL_SKILLS'] = ', '.join(skill_list[:3])

        # Ensure project descriptions are within limits
        for i in [1, 2]:
            desc_key = f'DESCRIPTION_OF_THE_SIDE_PROJECT_{i}'
            if desc_key in constrained:
                constrained[desc_key] = self._truncate_with_ellipsis(
                    constrained[desc_key],
                    CONTENT_LIMITS['project_description']
                )

        return constrained

    def _apply_cover_letter_constraints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply cover letter-specific single-page constraints to data.

        Args:
            data: Cover letter template data

        Returns:
            Constrained data optimized for cover letter single page
        """
        constrained = data.copy()

        # Apply paragraph length limits
        paragraph_fields = {
            'OPENING_PARAGRAPH': CONTENT_LIMITS['cover_letter_opening'],
            'EXPERIENCE_MATCH_PARAGRAPH': CONTENT_LIMITS['cover_letter_body_paragraph'],
            'SKILLS_PARAGRAPH': CONTENT_LIMITS['cover_letter_body_paragraph'],
            'CLOSING_PARAGRAPH': CONTENT_LIMITS['cover_letter_closing']
        }

        for field, limit in paragraph_fields.items():
            if field in constrained:
                constrained[field] = self._truncate_with_ellipsis(constrained[field], limit)

        return constrained

    def _calculate_content_metrics(self, content: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate content length metrics for space estimation.

        Args:
            content: Content dictionary

        Returns:
            Dictionary of content metrics
        """
        metrics = {}

        for key, value in content.items():
            if isinstance(value, str):
                metrics[key] = len(value)
            elif isinstance(value, (list, tuple)):
                metrics[key] = sum(len(str(item)) for item in value)
            else:
                metrics[key] = len(str(value))

        metrics['total_characters'] = sum(metrics.values())
        return metrics

    def _estimate_space_utilization(self, content_metrics: Dict[str, int]) -> float:
        """
        Estimate space utilization as percentage of single page.

        Args:
            content_metrics: Content length metrics

        Returns:
            Estimated space utilization (0.0-1.0)
        """
        total_chars = content_metrics.get('total_characters', 0)
        # Conservative estimate: 3500 characters for single page
        max_single_page_chars = 3500

        utilization = min(total_chars / max_single_page_chars, 1.0)
        return round(utilization, 3)

    def _validate_single_page_compliance(self, content_metrics: Dict[str, int]) -> bool:
        """
        Validate that content meets single-page constraints.

        Args:
            content_metrics: Content length metrics

        Returns:
            True if content should fit on single page
        """
        if not self.single_page_mode:
            return True

        total_chars = content_metrics.get('total_characters', 0)
        # Conservative limit for single page
        return total_chars <= 3500

    def _calculate_customization_score(self, match_result: MatchResult, content: Dict[str, Any]) -> float:
        """
        Calculate customization score based on match result and content personalization.

        Args:
            match_result: Job matching results
            content: Generated content

        Returns:
            Customization score (0.0-1.0)
        """
        score = 0.0

        # Base score from match result
        match_score_normalized = match_result.score / 10.0
        score += match_score_normalized * 0.4

        # Bonus for relevant experiences
        if len(match_result.relevant_experiences) > 0:
            score += 0.2

        # Bonus for recommended projects
        if len(match_result.recommended_projects) > 0:
            score += 0.2

        # Bonus for matched skills
        if len(match_result.matched_skills) > 0:
            score += 0.2

        return min(score, 1.0)