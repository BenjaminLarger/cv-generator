"""
ProfileMatcher class for comprehensive job-profile compatibility analysis using AI.

This module provides intelligent matching capabilities between user profiles and job
postings, leveraging OpenAI GPT-4 for nuanced analysis that goes beyond simple keyword
matching. It calculates detailed match scores, identifies relevant experiences, selects
optimal projects, and generates actionable improvement suggestions.
"""

import json
import os
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass

from openai import OpenAI
from pydantic import ValidationError as PydanticValidationError

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.job_data import JobData
from models.user_profile import UserProfile, WorkExperience, Project
from models.match_result import (
    MatchResult, MatchScore, AnalysisDetails, RelevantExperience,
    RecommendedProject, ImprovementSuggestion, SuggestionCategory
)
from utils.logging_config import get_agents_logger

logger = get_agents_logger()


class ProfileMatchingError(Exception):
    """Base exception for profile matching errors."""
    pass


class OpenAIConfigurationError(ProfileMatchingError):
    """Exception raised when OpenAI configuration is invalid."""
    pass


class MatchingAnalysisError(ProfileMatchingError):
    """Exception raised when matching analysis fails."""
    pass


class DataValidationError(ProfileMatchingError):
    """Exception raised when input data validation fails."""
    pass


@dataclass
class SkillMatchResult:
    """Result of skill matching analysis."""
    direct_matches: List[str]
    related_matches: List[Tuple[str, str, float]]  # (user_skill, job_skill, confidence)
    missing_skills: List[str]
    match_percentage: float
    transferable_skills: List[str]


class ProfileMatcher:
    """
    Comprehensive profile-job compatibility analyzer using AI.

    This class provides intelligent analysis of compatibility between user profiles
    and job postings. It uses OpenAI GPT-4 to perform nuanced analysis that considers
    not just keyword matching but context, relevance, and potential fit. The system
    provides detailed scoring, experience ranking, project recommendations, and
    actionable improvement suggestions.

    Attributes:
        openai_client: OpenAI client instance for API calls
        model: GPT model to use for analysis (default: gpt-4)
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay between retries in seconds

    Scoring Breakdown:
        - Skills match (40%): Direct + transferable skills alignment
        - Experience relevance (35%): Role alignment + achievements
        - Education fit (15%): Degree relevance + certifications
        - Project alignment (10%): Technology stack + impact
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the ProfileMatcher with OpenAI configuration.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: GPT model to use for analysis
            max_retries: Maximum retry attempts for API failures
            retry_delay: Base delay between retries in seconds

        Raises:
            OpenAIConfigurationError: If API key is not provided or invalid

        Example:
            >>> matcher = ProfileMatcher()
            >>> result = matcher.match_profile(job_data, user_profile)
            >>> print(f"Match score: {result.score}/10")
        """
        logger.info("Initializing ProfileMatcher")

        # Configure OpenAI client
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise OpenAIConfigurationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            self.openai_client = OpenAI(api_key=api_key)
            self.model = model
            self.max_retries = max_retries
            self.retry_delay = retry_delay

            logger.info(f"ProfileMatcher initialized with model: {model}, max_retries: {max_retries}")

        except Exception as e:
            raise OpenAIConfigurationError(f"Failed to initialize OpenAI client: {e}")

    def _call_openai_with_retry(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """
        Call OpenAI API with exponential backoff retry logic.

        Args:
            messages: List of message dictionaries for the chat completion
            temperature: Temperature setting for response consistency

        Returns:
            Response content from OpenAI

        Raises:
            MatchingAnalysisError: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"OpenAI API call attempt {attempt + 1}/{self.max_retries}")

                # Add delay for retries
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.debug(f"Waiting {delay} seconds before retry")
                    time.sleep(delay)

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=3000,
                    timeout=45
                )

                content = response.choices[0].message.content
                if not content:
                    raise MatchingAnalysisError("Empty response from OpenAI")

                logger.debug("Successfully received response from OpenAI")
                return content.strip()

            except Exception as e:
                last_exception = e
                logger.warning(f"OpenAI API call failed on attempt {attempt + 1}: {e}")

                # Don't retry for certain types of errors
                if any(err in str(e).lower() for err in ["invalid_api_key", "quota", "permission"]):
                    raise MatchingAnalysisError(f"OpenAI API error: {e}")

        # All retries exhausted
        error_msg = f"Failed to call OpenAI API after {self.max_retries} attempts"
        logger.error(error_msg)

        if last_exception:
            raise MatchingAnalysisError(f"{error_msg}: {last_exception}")
        else:
            raise MatchingAnalysisError(error_msg)

    def _create_matching_prompt(self, job_data: JobData, user_profile: UserProfile) -> List[Dict[str, str]]:
        """
        Create optimized prompts for comprehensive profile-job matching analysis.

        Args:
            job_data: Job posting data
            user_profile: User profile data

        Returns:
            List of messages for OpenAI chat completion
        """
        # Extract key profile information
        current_position = user_profile.get_current_position() or "Not currently employed"
        total_experience = user_profile.get_total_experience_years()
        skills_by_category = user_profile.get_skills_by_category()

        # Recent experiences (last 3)
        recent_experiences = user_profile.experiences[:3]

        # Most relevant projects (up to 4)
        recent_projects = user_profile.projects[:4]

        user_context = f"""
USER PROFILE SUMMARY:
Name: {user_profile.personal_info.name}
Current Position: {current_position}
Total Experience: {total_experience} years
Location: {user_profile.personal_info.location or 'Not specified'}

SKILLS ({len(user_profile.skills)} total):
Technical Skills: {', '.join(skills_by_category.get('technical', [])[:15])}
Soft Skills: {', '.join(skills_by_category.get('soft', [])[:10])}

RECENT WORK EXPERIENCE:
{self._format_experiences_for_prompt(recent_experiences)}

EDUCATION:
{self._format_education_for_prompt(user_profile.education)}

KEY PROJECTS:
{self._format_projects_for_prompt(recent_projects)}

PROFESSIONAL SUMMARY:
{user_profile.summary or 'No summary provided'}
"""

        job_context = f"""
JOB POSTING DETAILS:
Company: {job_data.company_name}
Position: {job_data.position}
Experience Level: {job_data.experience_level.upper()}

REQUIRED SKILLS ({len(job_data.skills_required)} total):
{', '.join(job_data.skills_required)}

REQUIREMENTS:
{self._format_requirements_for_prompt(job_data.requirements)}

JOB DESCRIPTION:
{job_data.description[:1000]}...
"""

        system_message = """You are an expert career consultant and talent acquisition specialist.
Analyze the compatibility between the user profile and job posting with deep insight into:
- Technical skill alignment and transferability
- Experience relevance and career progression
- Cultural and role fit indicators
- Growth potential and development opportunities

Provide nuanced analysis that goes beyond keyword matching to assess true compatibility."""

        user_message = f"""
Please analyze the compatibility between this user profile and job posting.

{user_context}

{job_context}

ANALYSIS REQUIREMENTS:
1. Calculate an overall match score (1-10) with detailed justification
2. Identify specific skills that match (direct and transferable)
3. Evaluate experience relevance with concrete examples
4. Assess education alignment
5. Consider cultural and role fit factors
6. Identify strengths and improvement areas
7. Provide actionable recommendations

SCORING CRITERIA:
- Skills Match (40%): Direct matches + transferable skills + learning potential
- Experience Relevance (35%): Role similarity + achievements + progression
- Education Fit (15%): Degree relevance + certifications + continuous learning
- Role Alignment (10%): Cultural fit + career goals + growth opportunity

Return your analysis in the following JSON format:
{{
    "overall_score": integer (1-10),
    "score_justification": "detailed explanation of the score",
    "skills_analysis": {{
        "direct_matches": ["list of directly matching skills"],
        "transferable_skills": ["skills that could transfer with explanation"],
        "missing_critical": ["skills that are critical but missing"],
        "match_percentage": float (0-100)
    }},
    "experience_analysis": {{
        "relevant_experiences": [
            {{
                "company": "company name",
                "role": "role title",
                "relevance_score": float (0.0-1.0),
                "relevance_reason": "why this experience is relevant",
                "key_achievements": ["most relevant achievements"]
            }}
        ],
        "experience_gap_assessment": "analysis of any experience gaps"
    }},
    "education_analysis": {{
        "education_match": boolean,
        "relevance_explanation": "how education aligns with job requirements"
    }},
    "strengths": ["key strength areas for this role"],
    "improvement_areas": ["specific areas needing development"],
    "recommendations": [
        {{
            "category": "skills|experience|education|projects|certifications|general",
            "priority": "high|medium|low",
            "suggestion": "specific actionable advice",
            "impact": "expected impact of following this advice"
        }}
    ],
    "confidence_level": float (0.0-1.0)
}}

Ensure all responses are actionable, specific, and provide concrete value for the user's career development.
"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def _format_experiences_for_prompt(self, experiences: List[WorkExperience]) -> str:
        """Format work experiences for prompt context."""
        if not experiences:
            return "No work experience provided"

        formatted = []
        for exp in experiences:
            duration = f"{exp.start_date.strftime('%Y-%m')}"
            if exp.end_date:
                duration += f" to {exp.end_date.strftime('%Y-%m')}"
            else:
                duration += " to Present"

            achievements = exp.achievements[:3]  # Top 3 achievements
            tech_stack = ', '.join(exp.technologies[:10])  # Limit technologies

            formatted.append(f"""
• {exp.role} at {exp.company} ({duration})
  Technologies: {tech_stack or 'Not specified'}
  Key Achievements: {'; '.join(achievements) if achievements else 'Not specified'}""")

        return '\n'.join(formatted)

    def _format_education_for_prompt(self, education: List) -> str:
        """Format education for prompt context."""
        if not education:
            return "No formal education provided"

        formatted = []
        for edu in education:
            year = f" ({edu.graduation_year})" if edu.graduation_year else ""
            field = f" in {edu.field_of_study}" if edu.field_of_study else ""
            honors = f" - {edu.honors}" if edu.honors else ""

            formatted.append(f"• {edu.degree}{field} from {edu.institution}{year}{honors}")

        return '\n'.join(formatted)

    def _format_projects_for_prompt(self, projects: List[Project]) -> str:
        """Format projects for prompt context."""
        if not projects:
            return "No projects provided"

        formatted = []
        for project in projects:
            tech_stack = ', '.join(project.technologies[:8])
            url = f" - {project.url}" if project.url else ""

            formatted.append(f"""
• {project.title}
  Technologies: {tech_stack or 'Not specified'}
  Description: {project.description[:200]}...{url}""")

        return '\n'.join(formatted)

    def _format_requirements_for_prompt(self, requirements: List[str]) -> str:
        """Format job requirements for prompt context."""
        if not requirements:
            return "No specific requirements listed"

        return '\n'.join(f"• {req}" for req in requirements[:10])  # Limit to 10 requirements

    def match_profile(self, job_data: JobData, user_profile: UserProfile) -> MatchResult:
        """
        Perform comprehensive profile-job compatibility analysis using AI.

        This method uses GPT-4 to analyze the compatibility between a user profile
        and job posting, providing detailed scoring, relevant experience identification,
        project recommendations, and actionable improvement suggestions.

        Args:
            job_data: JobData instance containing job posting information
            user_profile: UserProfile instance containing user's information

        Returns:
            MatchResult instance with comprehensive analysis results

        Raises:
            DataValidationError: If input data is invalid
            MatchingAnalysisError: If analysis fails

        Example:
            >>> matcher = ProfileMatcher()
            >>> job = JobData(company_name="TechCorp", position="Senior Developer", ...)
            >>> profile = UserProfile(personal_info=PersonalInfo(...), ...)
            >>> result = matcher.match_profile(job, profile)
            >>> print(f"Match: {result.score}/10 - {result.get_match_category()}")
        """
        logger.info(f"Starting profile matching analysis for {job_data.position} at {job_data.company_name}")

        # Validate inputs
        if not isinstance(job_data, JobData):
            raise DataValidationError("job_data must be a JobData instance")
        if not isinstance(user_profile, UserProfile):
            raise DataValidationError("user_profile must be a UserProfile instance")

        try:
            # Create analysis prompt
            messages = self._create_matching_prompt(job_data, user_profile)

            # Get AI analysis
            response_content = self._call_openai_with_retry(messages, temperature=0.1)

            # Parse JSON response
            try:
                analysis_data = json.loads(response_content)
                logger.debug("Successfully parsed AI analysis response")
            except json.JSONDecodeError as e:
                # Try to extract JSON if wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        analysis_data = json.loads(json_match.group())
                        logger.debug("Successfully extracted JSON from response")
                    except json.JSONDecodeError:
                        raise MatchingAnalysisError(f"Failed to parse AI response as JSON: {e}")
                else:
                    raise MatchingAnalysisError(f"No valid JSON found in AI response: {e}")

            # Extract and validate core data
            overall_score = analysis_data.get('overall_score', 5)
            if not (1 <= overall_score <= 10):
                logger.warning(f"Invalid score {overall_score}, defaulting to 5")
                overall_score = 5

            skills_analysis = analysis_data.get('skills_analysis', {})
            experience_analysis = analysis_data.get('experience_analysis', {})
            education_analysis = analysis_data.get('education_analysis', {})

            # Build relevant experiences
            relevant_experiences = self._build_relevant_experiences(
                experience_analysis.get('relevant_experiences', []),
                user_profile.experiences
            )

            # Select recommended projects
            recommended_projects = self.recommend_projects(user_profile.projects, job_data)

            # Build improvement suggestions
            suggestions = self._build_improvement_suggestions(
                analysis_data.get('recommendations', [])
            )

            # Create analysis details
            analysis_details = AnalysisDetails(
                skills_match_percentage=float(skills_analysis.get('match_percentage', 0.0)),
                experience_match_percentage=self._calculate_experience_match_percentage(relevant_experiences),
                education_match=bool(education_analysis.get('education_match', False)),
                missing_skills=skills_analysis.get('missing_critical', []),
                additional_skills=self._identify_additional_skills(user_profile.skills, job_data.skills_required),
                strength_areas=analysis_data.get('strengths', []),
                weakness_areas=analysis_data.get('improvement_areas', []),
                match_explanation=analysis_data.get('score_justification', ''),
                experience_gap_months=self._calculate_experience_gap(user_profile, job_data)
            )

            # Create match result
            match_result = MatchResult(
                score=MatchScore(overall_score),
                matched_skills=skills_analysis.get('direct_matches', []),
                relevant_experiences=relevant_experiences,
                recommended_projects=recommended_projects,
                suggestions=suggestions,
                analysis_details=analysis_details,
                job_title=job_data.position,
                company_name=job_data.company_name,
                confidence_level=float(analysis_data.get('confidence_level', 0.8))
            )

            logger.info(f"Profile matching completed: {match_result.score}/10 match for {job_data.position}")
            return match_result

        except (PydanticValidationError, KeyError, ValueError) as e:
            error_msg = f"Failed to create match result from analysis: {e}"
            logger.error(error_msg)
            raise MatchingAnalysisError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during profile matching: {e}"
            logger.error(error_msg)
            raise MatchingAnalysisError(error_msg)

    def calculate_skill_match(self, job_skills: List[str], user_skills: List[str]) -> Dict[str, Any]:
        """
        Perform detailed skill matching analysis with confidence scores.

        This method analyzes skill compatibility between job requirements and user
        skills, identifying direct matches, related/transferable skills, and missing
        critical skills with confidence scores.

        Args:
            job_skills: List of skills required by the job
            user_skills: List of skills possessed by the user

        Returns:
            Dictionary containing detailed skill match analysis:
            - direct_matches: List of exactly matching skills
            - related_matches: List of tuples (user_skill, job_skill, confidence)
            - missing_skills: List of job skills not covered
            - match_percentage: Overall percentage match
            - transferable_skills: Skills that could transfer to the role

        Example:
            >>> matcher = ProfileMatcher()
            >>> job_skills = ["Python", "Machine Learning", "AWS"]
            >>> user_skills = ["Python", "Deep Learning", "Docker"]
            >>> result = matcher.calculate_skill_match(job_skills, user_skills)
            >>> print(f"Match: {result['match_percentage']:.1f}%")
        """
        logger.debug(f"Analyzing skill match: {len(job_skills)} job skills vs {len(user_skills)} user skills")

        if not job_skills:
            return {
                "direct_matches": [],
                "related_matches": [],
                "missing_skills": [],
                "match_percentage": 100.0 if not user_skills else 0.0,
                "transferable_skills": user_skills[:10],  # Limited list
                "analysis_summary": "No job skills specified"
            }

        if not user_skills:
            return {
                "direct_matches": [],
                "related_matches": [],
                "missing_skills": job_skills,
                "match_percentage": 0.0,
                "transferable_skills": [],
                "analysis_summary": "No user skills provided"
            }

        # Normalize skills for comparison
        job_skills_normalized = [skill.lower().strip() for skill in job_skills]
        user_skills_normalized = [skill.lower().strip() for skill in user_skills]

        # Create skill mappings
        job_skill_map = {norm: orig for norm, orig in zip(job_skills_normalized, job_skills)}
        user_skill_map = {norm: orig for norm, orig in zip(user_skills_normalized, user_skills)}

        # Find direct matches
        direct_matches = []
        for job_skill_norm in job_skills_normalized:
            if job_skill_norm in user_skills_normalized:
                direct_matches.append(job_skill_map[job_skill_norm])

        # Find related/transferable skills using keyword matching
        related_matches = []
        skill_relationships = {
            # Programming languages
            'python': ['django', 'flask', 'fastapi', 'pandas', 'numpy'],
            'javascript': ['react', 'vue', 'angular', 'node.js', 'typescript'],
            'java': ['spring', 'hibernate', 'maven', 'gradle'],
            'c#': ['.net', 'asp.net', 'entity framework'],

            # Databases
            'sql': ['mysql', 'postgresql', 'sqlite', 'database'],
            'nosql': ['mongodb', 'cassandra', 'redis', 'dynamodb'],

            # Cloud platforms
            'aws': ['ec2', 'lambda', 's3', 'rds', 'cloudformation'],
            'azure': ['azure functions', 'azure sql', 'azure storage'],
            'gcp': ['google cloud', 'bigquery', 'compute engine'],

            # DevOps
            'docker': ['containerization', 'kubernetes', 'microservices'],
            'kubernetes': ['orchestration', 'container management'],
            'ci/cd': ['jenkins', 'github actions', 'gitlab ci'],

            # Data & ML
            'machine learning': ['deep learning', 'neural networks', 'ai', 'data science'],
            'data science': ['analytics', 'statistics', 'visualization'],
            'big data': ['hadoop', 'spark', 'kafka']
        }

        for job_skill_norm in job_skills_normalized:
            if job_skill_norm in direct_matches:
                continue

            for user_skill_norm in user_skills_normalized:
                confidence = self._calculate_skill_similarity(job_skill_norm, user_skill_norm, skill_relationships)
                if confidence > 0.3:  # Threshold for related skills
                    related_matches.append((
                        user_skill_map[user_skill_norm],
                        job_skill_map[job_skill_norm],
                        round(confidence, 2)
                    ))

        # Identify missing skills
        matched_job_skills = set(direct_matches + [match[1] for match in related_matches])
        missing_skills = [skill for skill in job_skills if skill not in matched_job_skills]

        # Calculate match percentage
        total_job_skills = len(job_skills)
        direct_weight = 1.0
        related_weight = 0.6

        total_score = (len(direct_matches) * direct_weight) + (len(related_matches) * related_weight)
        match_percentage = min(100.0, (total_score / total_job_skills) * 100)

        # Identify transferable skills (user skills not directly required but valuable)
        transferable_skills = []
        for user_skill in user_skills:
            if user_skill not in [match[0] for match in related_matches] and user_skill not in direct_matches:
                # Check if this skill could be valuable in the role context
                if any(keyword in user_skill.lower() for keyword in ['leadership', 'communication', 'project', 'agile', 'scrum']):
                    transferable_skills.append(user_skill)

        logger.debug(f"Skill analysis complete: {len(direct_matches)} direct, {len(related_matches)} related, {match_percentage:.1f}% match")

        return {
            "direct_matches": direct_matches,
            "related_matches": related_matches,
            "missing_skills": missing_skills,
            "match_percentage": round(match_percentage, 1),
            "transferable_skills": transferable_skills[:8],  # Limit to most relevant
            "analysis_summary": f"{len(direct_matches)} direct matches, {len(related_matches)} related skills found"
        }

    def _calculate_skill_similarity(self, job_skill: str, user_skill: str, relationships: Dict[str, List[str]]) -> float:
        """Calculate similarity confidence between two skills."""
        # Exact match
        if job_skill == user_skill:
            return 1.0

        # Substring match
        if job_skill in user_skill or user_skill in job_skill:
            return 0.8

        # Relationship-based matching
        for base_skill, related_skills in relationships.items():
            if base_skill in job_skill.lower():
                if user_skill.lower() in related_skills or any(rel in user_skill.lower() for rel in related_skills):
                    return 0.7
            if base_skill in user_skill.lower():
                if job_skill.lower() in related_skills or any(rel in job_skill.lower() for rel in related_skills):
                    return 0.6

        # Keyword overlap
        job_keywords = set(job_skill.lower().split())
        user_keywords = set(user_skill.lower().split())
        overlap = job_keywords.intersection(user_keywords)

        if overlap:
            return 0.4 + (len(overlap) / max(len(job_keywords), len(user_keywords))) * 0.3

        return 0.0

    def select_relevant_experiences(self, experiences: List[Dict], job_data: JobData) -> List[Dict]:
        """
        Rank and select the most relevant work experiences for a job application.

        This method analyzes work experiences to identify those most relevant to the
        job requirements, considering role similarity, technology stack alignment,
        achievement relevance, and career progression.

        Args:
            experiences: List of work experience dictionaries
            job_data: JobData instance containing job requirements

        Returns:
            List of relevant experience dictionaries sorted by relevance score

        Example:
            >>> experiences = [
            ...     {"company": "TechCorp", "role": "Senior Dev", "technologies": ["Python", "AWS"]},
            ...     {"company": "StartupCo", "role": "Full Stack Dev", "technologies": ["React", "Node.js"]}
            ... ]
            >>> relevant = matcher.select_relevant_experiences(experiences, job_data)
        """
        logger.debug(f"Selecting relevant experiences from {len(experiences)} total experiences")

        if not experiences:
            return []

        # Convert dict format to WorkExperience objects if needed
        work_experiences = []
        for exp in experiences:
            if isinstance(exp, dict):
                # Convert dict to WorkExperience-like object for processing
                work_experiences.append(exp)
            else:
                # Assume it's already a WorkExperience object
                work_experiences.append({
                    'company': getattr(exp, 'company', ''),
                    'role': getattr(exp, 'role', ''),
                    'technologies': getattr(exp, 'technologies', []),
                    'achievements': getattr(exp, 'achievements', []),
                    'start_date': getattr(exp, 'start_date', None),
                    'end_date': getattr(exp, 'end_date', None)
                })

        # Score each experience
        scored_experiences = []
        for exp in work_experiences:
            score = self._calculate_experience_relevance(exp, job_data)
            exp_with_score = {**exp, 'relevance_score': score}
            scored_experiences.append(exp_with_score)

        # Sort by relevance score (descending) and recency
        scored_experiences.sort(
            key=lambda x: (x['relevance_score'], x.get('start_date', date.min) if x.get('start_date') else date.min),
            reverse=True
        )

        # Return top relevant experiences (maximum 5)
        relevant_experiences = scored_experiences[:5]

        logger.debug(f"Selected {len(relevant_experiences)} relevant experiences")
        return relevant_experiences

    def _calculate_experience_relevance(self, experience: Dict, job_data: JobData) -> float:
        """Calculate relevance score for a work experience."""
        score = 0.0

        # Role similarity (30% weight)
        role_similarity = self._calculate_role_similarity(experience.get('role', ''), job_data.position)
        score += role_similarity * 0.30

        # Technology stack alignment (40% weight)
        tech_alignment = self._calculate_tech_alignment(
            experience.get('technologies', []),
            job_data.skills_required
        )
        score += tech_alignment * 0.40

        # Recency bonus (15% weight)
        recency_score = self._calculate_recency_score(experience.get('end_date'))
        score += recency_score * 0.15

        # Achievement relevance (15% weight)
        achievement_score = self._calculate_achievement_relevance(
            experience.get('achievements', []),
            job_data
        )
        score += achievement_score * 0.15

        return min(1.0, score)

    def _calculate_role_similarity(self, user_role: str, job_position: str) -> float:
        """Calculate similarity between user's role and job position."""
        if not user_role or not job_position:
            return 0.0

        user_role_lower = user_role.lower()
        job_position_lower = job_position.lower()

        # Exact match
        if user_role_lower == job_position_lower:
            return 1.0

        # Level indicators
        user_level = self._extract_seniority_level(user_role_lower)
        job_level = self._extract_seniority_level(job_position_lower)

        # Role type indicators
        common_role_keywords = ['developer', 'engineer', 'manager', 'analyst', 'architect', 'lead', 'senior', 'junior']
        user_keywords = set(word for word in user_role_lower.split() if word in common_role_keywords)
        job_keywords = set(word for word in job_position_lower.split() if word in common_role_keywords)

        keyword_overlap = len(user_keywords.intersection(job_keywords)) / max(len(job_keywords), 1)

        # Level alignment bonus
        level_alignment = 0.0
        if user_level == job_level:
            level_alignment = 0.3
        elif abs(user_level - job_level) == 1:
            level_alignment = 0.2

        return min(1.0, keyword_overlap + level_alignment)

    def _extract_seniority_level(self, role: str) -> int:
        """Extract seniority level from role title (0=intern, 1=junior, 2=mid, 3=senior, 4=lead, 5=executive)."""
        role_lower = role.lower()

        if any(word in role_lower for word in ['intern', 'trainee']):
            return 0
        elif any(word in role_lower for word in ['junior', 'associate', 'entry']):
            return 1
        elif any(word in role_lower for word in ['senior', 'sr.']):
            return 3
        elif any(word in role_lower for word in ['lead', 'principal', 'staff', 'architect']):
            return 4
        elif any(word in role_lower for word in ['director', 'vp', 'cto', 'ceo', 'head']):
            return 5
        else:
            return 2  # Mid-level default

    def _calculate_tech_alignment(self, user_technologies: List[str], job_skills: List[str]) -> float:
        """Calculate technology stack alignment score."""
        if not user_technologies or not job_skills:
            return 0.0

        user_tech_lower = [tech.lower().strip() for tech in user_technologies]
        job_skills_lower = [skill.lower().strip() for skill in job_skills]

        matches = sum(1 for tech in user_tech_lower if tech in job_skills_lower)

        return matches / len(job_skills_lower)

    def _calculate_recency_score(self, end_date: Optional[date]) -> float:
        """Calculate recency score for work experience."""
        if not end_date:  # Current job
            return 1.0

        today = date.today()
        years_ago = (today - end_date).days / 365.25

        if years_ago <= 1:
            return 1.0
        elif years_ago <= 3:
            return 0.8
        elif years_ago <= 5:
            return 0.6
        else:
            return 0.3

    def _calculate_achievement_relevance(self, achievements: List[str], job_data: JobData) -> float:
        """Calculate relevance score for achievements."""
        if not achievements:
            return 0.0

        relevant_keywords = (
            job_data.skills_required +
            [word.lower() for req in job_data.requirements for word in req.split()[:5]]  # Top words from requirements
        )
        relevant_keywords_lower = [kw.lower() for kw in relevant_keywords]

        relevance_scores = []
        for achievement in achievements:
            achievement_lower = achievement.lower()
            keyword_matches = sum(1 for kw in relevant_keywords_lower if kw in achievement_lower)
            relevance_scores.append(keyword_matches / len(relevant_keywords_lower) if relevant_keywords_lower else 0)

        return statistics.mean(relevance_scores) if relevance_scores else 0.0

    def recommend_projects(self, projects: List[Project], job_data: JobData) -> List[RecommendedProject]:
        """
        Select and recommend the most relevant projects for job application.

        This method analyzes user projects to identify those most relevant to the job
        requirements, considering technology stack alignment, project impact and scale,
        and relevance to the role. Maximum of 3-4 projects are recommended.

        Args:
            projects: List of Project instances from user profile
            job_data: JobData instance containing job requirements

        Returns:
            List of RecommendedProject instances sorted by relevance score

        Example:
            >>> projects = [project1, project2, project3, ...]
            >>> recommendations = matcher.recommend_projects(projects, job_data)
            >>> for proj in recommendations:
            ...     print(f"{proj.title}: {proj.relevance_score:.2f}")
        """
        logger.debug(f"Analyzing {len(projects)} projects for job relevance")

        if not projects:
            return []

        # Score each project
        scored_projects = []
        for project in projects:
            relevance_score = self._calculate_project_relevance(project, job_data)

            # Find matching technologies
            matching_tech = [
                tech for tech in project.technologies
                if any(tech.lower() in req_skill.lower() or req_skill.lower() in tech.lower()
                      for req_skill in job_data.skills_required)
            ]

            # Create reason for relevance
            reasons = []
            if matching_tech:
                reasons.append(f"Uses {', '.join(matching_tech[:3])} relevant to job requirements")

            if any(keyword in project.description.lower()
                  for keyword in ['scalable', 'performance', 'optimization', 'production']):
                reasons.append("Demonstrates production-ready development skills")

            if project.url:
                reasons.append("Has live demo/repository available")

            relevance_reason = '. '.join(reasons) if reasons else "Related to job domain and requirements"

            recommended_project = RecommendedProject(
                title=project.title,
                relevance_score=relevance_score,
                matching_technologies=matching_tech,
                description=project.description[:200] + "..." if len(project.description) > 200 else project.description,
                relevance_reason=relevance_reason,
                url=str(project.url) if project.url else None
            )

            scored_projects.append(recommended_project)

        # Sort by relevance score
        scored_projects.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top 3-4 most relevant projects
        top_projects = scored_projects[:4]

        logger.debug(f"Selected {len(top_projects)} recommended projects")
        return top_projects

    def _calculate_project_relevance(self, project: Project, job_data: JobData) -> float:
        """Calculate relevance score for a project."""
        score = 0.0

        # Technology alignment (60% weight)
        tech_score = self._calculate_project_tech_alignment(project.technologies, job_data.skills_required)
        score += tech_score * 0.60

        # Project complexity and impact (25% weight)
        complexity_score = self._assess_project_complexity(project)
        score += complexity_score * 0.25

        # Recency and status (15% weight)
        status_score = self._calculate_project_status_score(project)
        score += status_score * 0.15

        return min(1.0, score)

    def _calculate_project_tech_alignment(self, project_technologies: List[str], job_skills: List[str]) -> float:
        """Calculate how well project technologies align with job requirements."""
        if not project_technologies or not job_skills:
            return 0.0

        # Direct matches
        direct_matches = 0
        related_matches = 0

        for proj_tech in project_technologies:
            proj_tech_lower = proj_tech.lower()

            # Check for direct matches
            for job_skill in job_skills:
                job_skill_lower = job_skill.lower()

                if proj_tech_lower == job_skill_lower or proj_tech_lower in job_skill_lower or job_skill_lower in proj_tech_lower:
                    direct_matches += 1
                    break

                # Check for related technologies
                elif self._are_technologies_related(proj_tech_lower, job_skill_lower):
                    related_matches += 1
                    break

        # Calculate alignment score
        total_job_skills = len(job_skills)
        alignment_score = (direct_matches * 1.0 + related_matches * 0.6) / total_job_skills

        return min(1.0, alignment_score)

    def _are_technologies_related(self, tech1: str, tech2: str) -> bool:
        """Check if two technologies are related."""
        tech_families = [
            ['react', 'vue', 'angular', 'frontend', 'javascript'],
            ['python', 'django', 'flask', 'fastapi'],
            ['java', 'spring', 'hibernate'],
            ['aws', 'azure', 'gcp', 'cloud'],
            ['docker', 'kubernetes', 'containers'],
            ['mysql', 'postgresql', 'sql', 'database'],
            ['mongodb', 'nosql', 'redis'],
            ['machine learning', 'deep learning', 'ai', 'data science']
        ]

        for family in tech_families:
            if any(t in tech1 for t in family) and any(t in tech2 for t in family):
                return True

        return False

    def _assess_project_complexity(self, project: Project) -> float:
        """Assess project complexity and impact based on description and features."""
        description = project.description.lower()
        complexity_score = 0.3  # Base score

        # Complexity indicators
        complexity_keywords = [
            'scalable', 'microservices', 'api', 'database', 'authentication',
            'deployment', 'testing', 'ci/cd', 'production', 'performance',
            'optimization', 'architecture', 'distributed', 'real-time'
        ]

        keyword_matches = sum(1 for keyword in complexity_keywords if keyword in description)
        complexity_score += min(0.4, keyword_matches * 0.05)

        # Technology diversity (more technologies = more complex)
        tech_diversity = min(0.2, len(project.technologies) * 0.03)
        complexity_score += tech_diversity

        # URL presence (deployed/accessible project)
        if project.url:
            complexity_score += 0.1

        return min(1.0, complexity_score)

    def _calculate_project_status_score(self, project: Project) -> float:
        """Calculate score based on project status and recency."""
        if project.status.lower() == 'completed':
            status_score = 0.8
        elif project.status.lower() == 'ongoing':
            status_score = 1.0
        else:
            status_score = 0.5  # paused/archived

        # Recency bonus if dates are available
        if project.end_date or project.start_date:
            reference_date = project.end_date or project.start_date
            if reference_date:
                days_ago = (date.today() - reference_date).days
                if days_ago <= 365:  # Within last year
                    status_score += 0.2
                elif days_ago <= 730:  # Within last 2 years
                    status_score += 0.1

        return min(1.0, status_score)

    def generate_improvement_suggestions(self, match_result: MatchResult) -> List[ImprovementSuggestion]:
        """
        Generate actionable improvement suggestions based on match analysis.

        This method analyzes the match result to provide specific, actionable advice
        for strengthening the job application, including skill development recommendations,
        experience positioning tips, and application strategy guidance.

        Args:
            match_result: MatchResult instance from profile matching analysis

        Returns:
            List of ImprovementSuggestion instances prioritized by impact

        Example:
            >>> suggestions = matcher.generate_improvement_suggestions(match_result)
            >>> for suggestion in suggestions:
            ...     print(f"{suggestion.priority}: {suggestion.suggestion}")
        """
        logger.debug("Generating improvement suggestions from match analysis")

        suggestions = []
        details = match_result.analysis_details

        # Skills-based suggestions
        if details.missing_skills:
            high_priority_skills = details.missing_skills[:3]  # Top 3 missing skills
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.SKILLS,
                priority="high",
                suggestion=f"Develop proficiency in {', '.join(high_priority_skills)}. "
                          f"These are critical skills mentioned in the job requirements.",
                impact="Significantly improve your technical qualification for this role",
                resources=self._get_learning_resources(high_priority_skills)
            ))

        # Experience-based suggestions
        if details.experience_match_percentage < 70:
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.EXPERIENCE,
                priority="medium",
                suggestion="Highlight transferable experiences that demonstrate similar responsibilities or achievements. "
                          "Focus on quantifiable results and impact in your previous roles.",
                impact="Better positioning of your background to align with job requirements"
            ))

        # Education suggestions
        if not details.education_match and match_result.score <= 6:
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.EDUCATION,
                priority="low",
                suggestion="Consider relevant certifications or online courses to strengthen your educational background "
                          "for this role. Professional certifications can compensate for formal education gaps.",
                impact="Demonstrate commitment to professional development and fill knowledge gaps"
            ))

        # Project-based suggestions
        if len(match_result.recommended_projects) < 2:
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.PROJECTS,
                priority="medium",
                suggestion="Develop portfolio projects that showcase the key technologies and skills required for this role. "
                          "Focus on projects that solve real-world problems in this domain.",
                impact="Provide concrete evidence of your technical abilities and initiative"
            ))

        # Certification suggestions based on missing skills
        cert_skills = [skill for skill in details.missing_skills if self._requires_certification(skill)]
        if cert_skills:
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.CERTIFICATIONS,
                priority="medium",
                suggestion=f"Consider obtaining certifications in {', '.join(cert_skills[:2])}. "
                          f"These certifications are highly valued in the industry.",
                impact="Validate your expertise and demonstrate professional commitment",
                resources=self._get_certification_resources(cert_skills)
            ))

        # General application strategy
        if match_result.score <= 5:
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.GENERAL,
                priority="high",
                suggestion="Consider applying to roles that better match your current skill set while working on "
                          "developing the missing skills for this type of position.",
                impact="Increase your chances of success in the job market"
            ))
        elif match_result.score >= 8:
            suggestions.append(ImprovementSuggestion(
                category=SuggestionCategory.GENERAL,
                priority="low",
                suggestion="Your profile is a strong match! Focus on crafting a compelling cover letter that "
                          "highlights your most relevant experiences and achievements.",
                impact="Maximize your chances of getting an interview"
            ))

        logger.debug(f"Generated {len(suggestions)} improvement suggestions")
        return suggestions[:6]  # Limit to most important suggestions

    def _get_learning_resources(self, skills: List[str]) -> List[str]:
        """Get learning resources for specific skills."""
        resource_map = {
            'python': ['Python.org Official Tutorial', 'Codecademy Python Course', 'Real Python'],
            'javascript': ['MDN Web Docs', 'JavaScript.info', 'FreeCodeCamp'],
            'react': ['React Official Documentation', 'React Tutorial by Kent C. Dodds'],
            'aws': ['AWS Training and Certification', 'AWS Well-Architected Framework'],
            'docker': ['Docker Official Documentation', 'Docker Mastery Course'],
            'sql': ['W3Schools SQL Tutorial', 'SQLBolt Interactive Lessons'],
            'machine learning': ['Coursera ML Course by Andrew Ng', 'Fast.ai'],
        }

        resources = []
        for skill in skills[:2]:  # Limit resources
            skill_lower = skill.lower()
            for key, res_list in resource_map.items():
                if key in skill_lower:
                    resources.extend(res_list[:2])
                    break

        return resources[:4] if resources else ['Coursera', 'Udemy', 'LinkedIn Learning', 'YouTube tutorials']

    def _get_certification_resources(self, skills: List[str]) -> List[str]:
        """Get certification resources for specific skills."""
        cert_map = {
            'aws': ['AWS Certified Solutions Architect', 'AWS Certified Developer'],
            'azure': ['Microsoft Azure Fundamentals', 'Azure Developer Associate'],
            'google cloud': ['Google Cloud Professional Cloud Architect'],
            'pmp': ['Project Management Professional (PMP)'],
            'scrum': ['Certified ScrumMaster (CSM)'],
        }

        resources = []
        for skill in skills[:2]:
            skill_lower = skill.lower()
            for key, cert_list in cert_map.items():
                if key in skill_lower:
                    resources.extend(cert_list)
                    break

        return resources[:3]

    def _requires_certification(self, skill: str) -> bool:
        """Check if a skill typically requires certification."""
        cert_skills = ['aws', 'azure', 'gcp', 'pmp', 'scrum', 'cissp', 'comptia', 'cisco', 'oracle']
        return any(cert_skill in skill.lower() for cert_skill in cert_skills)

    # Helper methods for building results

    def _build_relevant_experiences(self, ai_experiences: List[Dict], user_experiences: List[WorkExperience]) -> List[RelevantExperience]:
        """Build RelevantExperience objects from AI analysis."""
        relevant_experiences = []

        for ai_exp in ai_experiences[:4]:  # Limit to top 4
            # Find matching experience in user profile
            matching_exp = None
            for user_exp in user_experiences:
                if (user_exp.company.lower() in ai_exp.get('company', '').lower() or
                    ai_exp.get('company', '').lower() in user_exp.company.lower()):
                    matching_exp = user_exp
                    break

            if matching_exp:
                relevant_experiences.append(RelevantExperience(
                    company=ai_exp.get('company', matching_exp.company),
                    role=ai_exp.get('role', matching_exp.role),
                    relevance_score=float(ai_exp.get('relevance_score', 0.5)),
                    matching_skills=ai_exp.get('matching_skills', matching_exp.technologies[:5]),
                    key_achievements=ai_exp.get('key_achievements', matching_exp.achievements[:3]),
                    duration_months=matching_exp.get_duration_months(),
                    relevance_reason=ai_exp.get('relevance_reason', 'Relevant experience for this role')
                ))

        return relevant_experiences

    def _build_improvement_suggestions(self, ai_recommendations: List[Dict]) -> List[ImprovementSuggestion]:
        """Build ImprovementSuggestion objects from AI analysis."""
        suggestions = []

        for rec in ai_recommendations:
            try:
                category = rec.get('category', 'general').lower()
                # Map category to enum
                category_enum = {
                    'skills': SuggestionCategory.SKILLS,
                    'experience': SuggestionCategory.EXPERIENCE,
                    'education': SuggestionCategory.EDUCATION,
                    'projects': SuggestionCategory.PROJECTS,
                    'certifications': SuggestionCategory.CERTIFICATIONS,
                    'general': SuggestionCategory.GENERAL
                }.get(category, SuggestionCategory.GENERAL)

                suggestions.append(ImprovementSuggestion(
                    category=category_enum,
                    priority=rec.get('priority', 'medium').lower(),
                    suggestion=rec.get('suggestion', 'General improvement recommendation'),
                    impact=rec.get('impact', 'Positive impact on application strength')
                ))
            except Exception as e:
                logger.warning(f"Failed to parse recommendation: {e}")
                continue

        return suggestions

    def _calculate_experience_match_percentage(self, relevant_experiences: List[RelevantExperience]) -> float:
        """Calculate experience match percentage from relevant experiences."""
        if not relevant_experiences:
            return 0.0

        # Average relevance score of top experiences
        scores = [exp.relevance_score for exp in relevant_experiences]
        return statistics.mean(scores) * 100

    def _identify_additional_skills(self, user_skills: List[str], job_skills: List[str]) -> List[str]:
        """Identify user skills not required but potentially valuable."""
        job_skills_lower = [skill.lower() for skill in job_skills]
        additional_skills = []

        for user_skill in user_skills:
            if user_skill.lower() not in job_skills_lower:
                # Check if it's a valuable transferable skill
                if any(keyword in user_skill.lower() for keyword in
                      ['leadership', 'management', 'communication', 'agile', 'project']):
                    additional_skills.append(user_skill)

        return additional_skills[:8]  # Limit to most relevant

    def _calculate_experience_gap(self, user_profile: UserProfile, job_data: JobData) -> Optional[int]:
        """Calculate experience gap in months based on job requirements."""
        user_experience_years = user_profile.get_total_experience_years()

        # Extract experience requirement from job posting
        experience_keywords = ['years', 'experience', 'minimum']
        required_experience = 0

        for req in job_data.requirements:
            req_lower = req.lower()
            if any(keyword in req_lower for keyword in experience_keywords):
                # Simple pattern matching for experience requirements
                import re
                numbers = re.findall(r'\d+', req)
                if numbers:
                    required_experience = max(required_experience, int(numbers[0]))

        # Use experience level as fallback
        if required_experience == 0:
            level_map = {
                'entry': 0, 'junior': 2, 'mid': 4,
                'senior': 6, 'lead': 8, 'executive': 10
            }
            required_experience = level_map.get(job_data.experience_level.value, 4)

        gap_years = max(0, required_experience - user_experience_years)
        return int(gap_years * 12) if gap_years > 0 else None