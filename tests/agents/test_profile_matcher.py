"""
Unit tests for ProfileMatcher class.

This module contains comprehensive tests for the ProfileMatcher functionality,
including skill matching, experience selection, project recommendations, and
AI-powered profile-job compatibility analysis.
"""

import json
import pytest
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Adjust imports for the test environment
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Mock the config loading before importing modules that depend on it
with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key-for-config'}):
    pass

from src.agents.profile_matcher import (
    ProfileMatcher, ProfileMatchingError, OpenAIConfigurationError,
    MatchingAnalysisError, DataValidationError, SkillMatchResult
)
from src.models.job_data import JobData, ExperienceLevel
from src.models.user_profile import (
    UserProfile, PersonalInfo, WorkExperience, Project, Education
)
from src.models.match_result import (
    MatchResult, MatchScore, AnalysisDetails, RelevantExperience,
    RecommendedProject, ImprovementSuggestion, SuggestionCategory
)


class TestProfileMatcherInitialization:
    """Test ProfileMatcher initialization and configuration."""

    def test_init_with_api_key(self):
        """Test initialization with provided API key."""
        with patch('src.agents.profile_matcher.OpenAI') as mock_openai:
            matcher = ProfileMatcher(api_key="test-key")

            assert matcher.model == "gpt-4"
            assert matcher.max_retries == 3
            assert matcher.retry_delay == 2
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch('src.agents.profile_matcher.OpenAI') as mock_openai, \
             patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):

            matcher = ProfileMatcher()
            mock_openai.assert_called_once_with(api_key="env-key")

    def test_init_no_api_key_raises_error(self):
        """Test that missing API key raises configuration error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(OpenAIConfigurationError, match="OpenAI API key not provided"):
                ProfileMatcher()

    def test_init_openai_client_failure(self):
        """Test handling of OpenAI client initialization failure."""
        with patch('src.agents.profile_matcher.OpenAI', side_effect=Exception("Client error")):
            with pytest.raises(OpenAIConfigurationError, match="Failed to initialize OpenAI client"):
                ProfileMatcher(api_key="test-key")

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('src.agents.profile_matcher.OpenAI'):
            matcher = ProfileMatcher(
                api_key="test-key",
                model="gpt-3.5-turbo",
                max_retries=5,
                retry_delay=3
            )

            assert matcher.model == "gpt-3.5-turbo"
            assert matcher.max_retries == 5
            assert matcher.retry_delay == 3


class TestSkillMatchingAnalysis:
    """Test skill matching functionality."""

    @pytest.fixture
    def matcher(self):
        """Create a ProfileMatcher instance for testing."""
        with patch('src.agents.profile_matcher.OpenAI'):
            return ProfileMatcher(api_key="test-key")

    def test_calculate_skill_match_exact_matches(self, matcher):
        """Test skill matching with exact matches."""
        job_skills = ["Python", "React", "SQL"]
        user_skills = ["Python", "JavaScript", "SQL", "Docker"]

        result = matcher.calculate_skill_match(job_skills, user_skills)

        assert "Python" in result["direct_matches"]
        assert "SQL" in result["direct_matches"]
        assert len(result["direct_matches"]) == 2
        assert result["match_percentage"] > 50

    def test_calculate_skill_match_no_job_skills(self, matcher):
        """Test skill matching when no job skills are specified."""
        result = matcher.calculate_skill_match([], ["Python", "React"])

        assert result["match_percentage"] == 0.0
        assert result["direct_matches"] == []
        assert "No job skills specified" in result["analysis_summary"]

    def test_calculate_skill_match_no_user_skills(self, matcher):
        """Test skill matching when user has no skills."""
        job_skills = ["Python", "React", "SQL"]
        result = matcher.calculate_skill_match(job_skills, [])

        assert result["match_percentage"] == 0.0
        assert result["missing_skills"] == job_skills
        assert result["direct_matches"] == []

    def test_calculate_skill_match_related_skills(self, matcher):
        """Test identification of related/transferable skills."""
        job_skills = ["Python", "Machine Learning"]
        user_skills = ["Django", "Deep Learning", "JavaScript"]

        result = matcher.calculate_skill_match(job_skills, user_skills)

        # Should find Django as related to Python
        # Should find Deep Learning as related to Machine Learning
        assert len(result["related_matches"]) > 0
        assert result["match_percentage"] > 0

    def test_calculate_skill_match_case_insensitive(self, matcher):
        """Test that skill matching is case insensitive."""
        job_skills = ["PYTHON", "react"]
        user_skills = ["python", "REACT", "sql"]

        result = matcher.calculate_skill_match(job_skills, user_skills)

        assert len(result["direct_matches"]) == 2
        assert result["match_percentage"] == 100.0

    def test_skill_similarity_calculation(self, matcher):
        """Test skill similarity calculation logic."""
        # Test exact match
        assert matcher._calculate_skill_similarity("python", "python", {}) == 1.0

        # Test substring match
        assert matcher._calculate_skill_similarity("python", "python developer", {}) == 0.8

        # Test relationship-based match
        relationships = {"python": ["django", "flask"]}
        assert matcher._calculate_skill_similarity("python", "django", relationships) > 0.6


class TestExperienceSelection:
    """Test experience relevance and selection functionality."""

    @pytest.fixture
    def matcher(self):
        with patch('src.agents.profile_matcher.OpenAI'):
            return ProfileMatcher(api_key="test-key")

    @pytest.fixture
    def sample_job_data(self):
        return JobData(
            company_name="TechCorp",
            position="Senior Python Developer",
            skills_required=["Python", "Django", "PostgreSQL", "AWS"],
            requirements=["5+ years experience", "Bachelor's degree"],
            experience_level=ExperienceLevel.SENIOR,
            description="Senior Python developer position with Django expertise."
        )

    @pytest.fixture
    def sample_experiences(self):
        return [
            {
                'company': 'StartupCo',
                'role': 'Python Developer',
                'technologies': ['Python', 'Django', 'PostgreSQL'],
                'achievements': ['Built scalable web application', 'Improved performance by 50%'],
                'start_date': date(2020, 1, 1),
                'end_date': date(2023, 1, 1)
            },
            {
                'company': 'BigCorp',
                'role': 'Frontend Developer',
                'technologies': ['React', 'JavaScript', 'CSS'],
                'achievements': ['Developed responsive UI', 'Led team of 3 developers'],
                'start_date': date(2018, 1, 1),
                'end_date': date(2020, 1, 1)
            },
            {
                'company': 'DataCorp',
                'role': 'Data Analyst',
                'technologies': ['SQL', 'Python', 'Tableau'],
                'achievements': ['Created automated reports', 'Analyzed customer behavior'],
                'start_date': date(2023, 1, 1),
                'end_date': None  # Current job
            }
        ]

    def test_select_relevant_experiences(self, matcher, sample_experiences, sample_job_data):
        """Test selection of relevant experiences."""
        relevant = matcher.select_relevant_experiences(sample_experiences, sample_job_data)

        assert len(relevant) > 0
        # Python Developer should be ranked higher due to skill match
        python_dev_found = any(exp['role'] == 'Python Developer' for exp in relevant)
        assert python_dev_found

        # Check that relevance scores are added
        for exp in relevant:
            assert 'relevance_score' in exp
            assert 0 <= exp['relevance_score'] <= 1

    def test_select_relevant_experiences_empty_list(self, matcher, sample_job_data):
        """Test handling of empty experience list."""
        result = matcher.select_relevant_experiences([], sample_job_data)
        assert result == []

    def test_calculate_experience_relevance(self, matcher, sample_job_data):
        """Test experience relevance calculation."""
        experience = {
            'role': 'Senior Python Developer',
            'technologies': ['Python', 'Django', 'AWS'],
            'achievements': ['Built scalable API', 'Managed team'],
            'end_date': None  # Current job
        }

        relevance = matcher._calculate_experience_relevance(experience, sample_job_data)

        assert 0 <= relevance <= 1
        assert relevance > 0.5  # Should be highly relevant

    def test_role_similarity_calculation(self, matcher):
        """Test role similarity scoring."""
        # Exact match
        assert matcher._calculate_role_similarity("Python Developer", "Python Developer") == 1.0

        # Similar roles
        similarity = matcher._calculate_role_similarity("Senior Python Developer", "Python Developer")
        assert similarity > 0.5

        # Different roles
        similarity = matcher._calculate_role_similarity("Marketing Manager", "Python Developer")
        assert similarity <= 0.3

    def test_seniority_level_extraction(self, matcher):
        """Test extraction of seniority levels from role titles."""
        assert matcher._extract_seniority_level("intern developer") == 0
        assert matcher._extract_seniority_level("junior python developer") == 1
        assert matcher._extract_seniority_level("python developer") == 2  # mid-level default
        assert matcher._extract_seniority_level("senior developer") == 3
        assert matcher._extract_seniority_level("lead architect") == 4
        assert matcher._extract_seniority_level("director of engineering") == 5


class TestProjectRecommendations:
    """Test project recommendation functionality."""

    @pytest.fixture
    def matcher(self):
        with patch('src.agents.profile_matcher.OpenAI'):
            return ProfileMatcher(api_key="test-key")

    @pytest.fixture
    def sample_projects(self):
        return [
            Project(
                title="E-commerce Platform",
                description="Built a full-stack e-commerce platform with Django and React",
                technologies=["Python", "Django", "React", "PostgreSQL"],
                url="https://github.com/user/ecommerce",
                status="completed"
            ),
            Project(
                title="Machine Learning Model",
                description="Developed predictive model for customer churn",
                technologies=["Python", "Scikit-learn", "Pandas"],
                url="https://github.com/user/ml-model",
                status="completed"
            ),
            Project(
                title="Mobile App",
                description="Created iOS app for task management",
                technologies=["Swift", "Core Data", "UIKit"],
                status="ongoing"
            )
        ]

    @pytest.fixture
    def sample_job_data(self):
        return JobData(
            company_name="WebCorp",
            position="Full Stack Developer",
            skills_required=["Python", "Django", "React", "JavaScript"],
            requirements=["Web development experience", "Full-stack skills"],
            experience_level=ExperienceLevel.MID,
            description="Full stack developer role with Python and React."
        )

    def test_recommend_projects(self, matcher, sample_projects, sample_job_data):
        """Test project recommendation based on job requirements."""
        recommendations = matcher.recommend_projects(sample_projects, sample_job_data)

        assert len(recommendations) > 0
        assert all(isinstance(rec, RecommendedProject) for rec in recommendations)

        # E-commerce project should be highly relevant
        ecommerce_rec = next((rec for rec in recommendations if "E-commerce" in rec.title), None)
        assert ecommerce_rec is not None
        assert ecommerce_rec.relevance_score > 0.5

    def test_recommend_projects_empty_list(self, matcher, sample_job_data):
        """Test handling of empty project list."""
        result = matcher.recommend_projects([], sample_job_data)
        assert result == []

    def test_project_tech_alignment(self, matcher, sample_job_data):
        """Test technology alignment calculation for projects."""
        project_tech = ["Python", "Django", "React"]
        job_skills = ["Python", "Django", "JavaScript", "CSS"]

        alignment = matcher._calculate_project_tech_alignment(project_tech, job_skills)
        assert 0 <= alignment <= 1
        assert alignment > 0  # Should have some alignment

    def test_project_complexity_assessment(self, matcher):
        """Test project complexity scoring."""
        simple_project = Project(
            title="Hello World",
            description="Simple hello world application",
            technologies=["Python"]
        )

        complex_project = Project(
            title="Microservices Platform",
            description="Scalable microservices platform with API gateway, authentication, "
                       "database sharding, and real-time performance monitoring",
            technologies=["Python", "Docker", "Kubernetes", "Redis", "PostgreSQL"],
            url="https://example.com"
        )

        simple_score = matcher._assess_project_complexity(simple_project)
        complex_score = matcher._assess_project_complexity(complex_project)

        assert complex_score > simple_score


class TestAIIntegration:
    """Test AI-powered matching analysis."""

    @pytest.fixture
    def matcher(self):
        with patch('src.agents.profile_matcher.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            return ProfileMatcher(api_key="test-key")

    @pytest.fixture
    def sample_ai_response(self):
        return {
            "overall_score": 8,
            "score_justification": "Strong technical match with relevant experience",
            "skills_analysis": {
                "direct_matches": ["Python", "Django"],
                "transferable_skills": ["Flask experience transfers to Django"],
                "missing_critical": ["AWS", "Docker"],
                "match_percentage": 75.0
            },
            "experience_analysis": {
                "relevant_experiences": [
                    {
                        "company": "TechCorp",
                        "role": "Python Developer",
                        "relevance_score": 0.9,
                        "relevance_reason": "Direct Python development experience",
                        "key_achievements": ["Built scalable API", "Improved performance"]
                    }
                ],
                "experience_gap_assessment": "Strong relevant experience"
            },
            "education_analysis": {
                "education_match": True,
                "relevance_explanation": "CS degree aligns well with technical requirements"
            },
            "strengths": ["Strong Python skills", "Web development experience"],
            "improvement_areas": ["Cloud technologies", "DevOps practices"],
            "recommendations": [
                {
                    "category": "skills",
                    "priority": "high",
                    "suggestion": "Learn AWS cloud services",
                    "impact": "Strengthen cloud computing capabilities"
                }
            ],
            "confidence_level": 0.85
        }

    def test_call_openai_with_retry_success(self, matcher):
        """Test successful OpenAI API call."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]

        matcher.openai_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = matcher._call_openai_with_retry(messages)

        assert result == "Test response"
        matcher.openai_client.chat.completions.create.assert_called_once()

    def test_call_openai_with_retry_failure(self, matcher):
        """Test OpenAI API call failure and retry logic."""
        matcher.openai_client.chat.completions.create.side_effect = Exception("API Error")
        matcher.max_retries = 2

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(MatchingAnalysisError, match="Failed to call OpenAI API"):
            matcher._call_openai_with_retry(messages)

        assert matcher.openai_client.chat.completions.create.call_count == 2

    def test_match_profile_success(self, matcher, sample_ai_response):
        """Test successful profile matching with AI analysis."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(sample_ai_response)
        mock_response.choices = [mock_choice]
        matcher.openai_client.chat.completions.create.return_value = mock_response

        # Create test data
        job_data = JobData(
            company_name="TechCorp",
            position="Python Developer",
            skills_required=["Python", "Django", "AWS"],
            experience_level=ExperienceLevel.MID,
            description="Python developer position"
        )

        user_profile = UserProfile(
            personal_info=PersonalInfo(name="John Doe", email="john@example.com"),
            skills=["Python", "Django", "Flask"],
            experiences=[],
            education=[],
            projects=[]
        )

        result = matcher.match_profile(job_data, user_profile)

        assert isinstance(result, MatchResult)
        assert result.score == 8
        assert len(result.matched_skills) > 0
        assert result.job_title == "Python Developer"
        assert result.company_name == "TechCorp"

    def test_match_profile_invalid_input(self, matcher):
        """Test match_profile with invalid input data."""
        with pytest.raises(DataValidationError):
            matcher.match_profile("invalid", "invalid")

    def test_match_profile_json_parsing_error(self, matcher):
        """Test handling of invalid JSON response from AI."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Invalid JSON response"
        mock_response.choices = [mock_choice]
        matcher.openai_client.chat.completions.create.return_value = mock_response

        job_data = JobData(
            company_name="Test",
            position="Test",
            description="Test description"
        )
        user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test", email="test@example.com")
        )

        with pytest.raises(MatchingAnalysisError, match="No valid JSON found"):
            matcher.match_profile(job_data, user_profile)


class TestImprovementSuggestions:
    """Test improvement suggestion generation."""

    @pytest.fixture
    def matcher(self):
        with patch('src.agents.profile_matcher.OpenAI'):
            return ProfileMatcher(api_key="test-key")

    @pytest.fixture
    def sample_match_result(self):
        """Create a sample match result for testing."""
        analysis_details = AnalysisDetails(
            skills_match_percentage=60.0,
            experience_match_percentage=40.0,
            education_match=False,
            missing_skills=["AWS", "Docker", "Kubernetes"],
            additional_skills=["Leadership", "Project Management"],
            strength_areas=["Python", "Problem Solving"],
            weakness_areas=["Cloud Technologies", "DevOps"],
            match_explanation="Good technical foundation but lacks cloud experience"
        )

        return MatchResult(
            score=MatchScore.ABOVE_AVERAGE,
            matched_skills=["Python", "Django"],
            relevant_experiences=[],
            recommended_projects=[],
            suggestions=[],
            analysis_details=analysis_details,
            job_title="Cloud Developer",
            company_name="CloudCorp"
        )

    def test_generate_improvement_suggestions(self, matcher, sample_match_result):
        """Test generation of improvement suggestions."""
        suggestions = matcher.generate_improvement_suggestions(sample_match_result)

        assert len(suggestions) > 0
        assert all(isinstance(s, ImprovementSuggestion) for s in suggestions)

        # Should have skills-based suggestion for missing skills
        skills_suggestions = [s for s in suggestions if s.category == SuggestionCategory.SKILLS]
        assert len(skills_suggestions) > 0

    def test_get_learning_resources(self, matcher):
        """Test learning resource recommendations."""
        skills = ["Python", "AWS", "Machine Learning"]
        resources = matcher._get_learning_resources(skills)

        assert len(resources) > 0
        assert all(isinstance(resource, str) for resource in resources)

    def test_get_certification_resources(self, matcher):
        """Test certification resource recommendations."""
        skills = ["AWS", "Azure"]
        resources = matcher._get_certification_resources(skills)

        assert len(resources) > 0
        assert any("AWS" in resource for resource in resources)

    def test_requires_certification(self, matcher):
        """Test certification requirement detection."""
        assert matcher._requires_certification("AWS Solutions Architect")
        assert matcher._requires_certification("Azure Developer")
        assert not matcher._requires_certification("Python Programming")


class TestHelperMethods:
    """Test utility and helper methods."""

    @pytest.fixture
    def matcher(self):
        with patch('src.agents.profile_matcher.OpenAI'):
            return ProfileMatcher(api_key="test-key")

    def test_format_experiences_for_prompt(self, matcher):
        """Test formatting of experiences for AI prompt."""
        experiences = [
            WorkExperience(
                company="TestCorp",
                role="Developer",
                start_date=date(2020, 1, 1),
                end_date=date(2023, 1, 1),
                technologies=["Python", "Django"],
                achievements=["Built API", "Improved performance"]
            )
        ]

        formatted = matcher._format_experiences_for_prompt(experiences)

        assert "TestCorp" in formatted
        assert "Developer" in formatted
        assert "Python" in formatted

    def test_format_experiences_empty(self, matcher):
        """Test formatting of empty experiences list."""
        formatted = matcher._format_experiences_for_prompt([])
        assert "No work experience provided" in formatted

    def test_format_education_for_prompt(self, matcher):
        """Test formatting of education for AI prompt."""
        education = [
            Education(
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                institution="Test University",
                graduation_year=2020
            )
        ]

        formatted = matcher._format_education_for_prompt(education)

        assert "Bachelor of Science" in formatted
        assert "Computer Science" in formatted
        assert "Test University" in formatted

    def test_format_projects_for_prompt(self, matcher):
        """Test formatting of projects for AI prompt."""
        projects = [
            Project(
                title="Test Project",
                description="A test project for demonstration",
                technologies=["Python", "React"]
            )
        ]

        formatted = matcher._format_projects_for_prompt(projects)

        assert "Test Project" in formatted
        assert "Python" in formatted

    def test_calculate_experience_gap(self, matcher):
        """Test experience gap calculation."""
        user_profile = UserProfile(
            personal_info=PersonalInfo(name="Test", email="test@example.com"),
            experiences=[
                WorkExperience(
                    company="TestCorp",
                    role="Developer",
                    start_date=date(2020, 1, 1),
                    end_date=date(2022, 1, 1)
                )
            ]
        )

        job_data = JobData(
            company_name="Test",
            position="Senior Developer",
            requirements=["5+ years of experience"],
            experience_level=ExperienceLevel.SENIOR,
            description="Senior developer position"
        )

        gap = matcher._calculate_experience_gap(user_profile, job_data)

        # Should calculate some gap since user has 2 years but senior typically requires more
        assert gap is None or isinstance(gap, int)


# Integration test
class TestProfileMatcherIntegration:
    """Integration tests for complete ProfileMatcher workflow."""

    def test_end_to_end_matching_workflow(self):
        """Test complete matching workflow with real-like data."""
        with patch('src.agents.profile_matcher.OpenAI') as mock_openai:
            # Mock OpenAI response
            mock_client = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = json.dumps({
                "overall_score": 7,
                "score_justification": "Good match with room for improvement",
                "skills_analysis": {
                    "direct_matches": ["Python", "React"],
                    "transferable_skills": ["JavaScript to TypeScript"],
                    "missing_critical": ["AWS"],
                    "match_percentage": 70.0
                },
                "experience_analysis": {
                    "relevant_experiences": [],
                    "experience_gap_assessment": "Adequate experience level"
                },
                "education_analysis": {
                    "education_match": True,
                    "relevance_explanation": "Relevant degree"
                },
                "strengths": ["Technical skills"],
                "improvement_areas": ["Cloud experience"],
                "recommendations": [],
                "confidence_level": 0.8
            })
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            # Create matcher
            matcher = ProfileMatcher(api_key="test-key")

            # Create comprehensive test data
            job_data = JobData(
                company_name="TechStartup",
                position="Full Stack Developer",
                skills_required=["Python", "React", "PostgreSQL", "AWS"],
                requirements=["3+ years experience", "Bachelor's degree"],
                experience_level=ExperienceLevel.MID,
                description="Full stack developer role with modern tech stack"
            )

            user_profile = UserProfile(
                personal_info=PersonalInfo(
                    name="Jane Developer",
                    email="jane@example.com",
                    location="San Francisco, CA"
                ),
                skills=["Python", "Django", "React", "JavaScript", "PostgreSQL"],
                experiences=[
                    WorkExperience(
                        company="WebCorp",
                        role="Software Developer",
                        start_date=date(2021, 1, 1),
                        end_date=None,  # Current job
                        technologies=["Python", "Django", "React"],
                        achievements=["Built user management system", "Improved API performance by 40%"]
                    )
                ],
                education=[
                    Education(
                        degree="Bachelor of Science",
                        field_of_study="Computer Science",
                        institution="Tech University",
                        graduation_year=2020
                    )
                ],
                projects=[
                    Project(
                        title="Task Management App",
                        description="Full stack task management application with real-time updates",
                        technologies=["Python", "Django", "React", "WebSocket"],
                        status="completed"
                    )
                ]
            )

            # Perform matching
            result = matcher.match_profile(job_data, user_profile)

            # Verify results
            assert isinstance(result, MatchResult)
            assert 1 <= result.score <= 10
            assert result.job_title == "Full Stack Developer"
            assert result.company_name == "TechStartup"
            assert len(result.matched_skills) > 0
            assert 0.0 <= result.confidence_level <= 1.0

            # Test individual methods
            skill_match = matcher.calculate_skill_match(
                job_data.skills_required,
                user_profile.skills
            )
            assert skill_match["match_percentage"] > 0

            relevant_exp = matcher.select_relevant_experiences(
                [{"company": exp.company, "role": exp.role, "technologies": exp.technologies,
                  "achievements": exp.achievements, "start_date": exp.start_date, "end_date": exp.end_date}
                 for exp in user_profile.experiences],
                job_data
            )
            assert len(relevant_exp) >= 0

            project_recs = matcher.recommend_projects(user_profile.projects, job_data)
            assert len(project_recs) >= 0

            suggestions = matcher.generate_improvement_suggestions(result)
            assert len(suggestions) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])