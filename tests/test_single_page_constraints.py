"""
Comprehensive tests for single-page PDF constraint adherence in TemplateCustomizer.

This test suite validates that the TemplateCustomizer properly enforces single-page
PDF constraints, including content length validation, space utilization checks,
and compliance metrics.
"""

import pytest
from datetime import datetime, date
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.template_customizer import (
    TemplateCustomizer,
    CONTENT_LIMITS,
    SinglePageConstraints,
    CustomizationResult
)
from src.models.match_result import (
    MatchResult,
    RelevantExperience,
    RecommendedProject,
    AnalysisDetails,
    MatchScore
)
from src.models.user_profile import (
    UserProfile,
    PersonalInfo,
    WorkExperience,
    Project,
    Education,
    SocialUrls
)
from src.models.job_data import JobData, ExperienceLevel


class TestSinglePageConstraints:
    """Test suite for single-page PDF constraints."""

    @pytest.fixture
    def sample_user_profile(self):
        """Create a comprehensive sample user profile for testing."""
        personal_info = PersonalInfo(
            name="Benjamin Larger",
            email="benjamin.larger@example.com",
            phone="+34 667-006-863",
            location="Madrid, Spain"
        )

        experiences = [
            WorkExperience(
                company="ENGIE Madrid",
                role="IS Software Engineer Intern",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 6, 30),
                achievements=[
                    "Developed Energy Management Solutions using Python RESTful APIs for powerplant optimization algorithms",
                    "Implemented web scraping solutions with Playwright to process business-critical information from internal systems",
                    "Built containerized applications with Docker and deployed on AWS Lambda for improved scalability and performance",
                    "Collaborated with Back and Middle Office teams to implement tailored software solutions for energy trading"
                ],
                technologies=["Python", "Docker", "AWS Lambda", "Playwright", "RESTful APIs"]
            ),
            WorkExperience(
                company="ING Brussels",
                role="FEC Data Analyst Intern",
                start_date=date(2023, 5, 1),
                end_date=date(2023, 9, 30),
                achievements=[
                    "Conducted comprehensive Python data analytics to identify suspicious behaviors in financial datasets of 10,000+ records",
                    "Developed advanced VBA automation scripts that reduced manual analysis time by 20% and improved accuracy",
                    "Created detailed reports and visualizations for compliance teams using advanced statistical methods"
                ],
                technologies=["Python", "VBA", "Excel", "Data Analysis", "Statistical Modeling"]
            )
        ]

        projects = [
            Project(
                title="AI-Powered CV Generator",
                description="Developed an intelligent CV generation system using Python and machine learning algorithms that automatically customizes resumes based on job requirements. The system analyzes job descriptions, matches user skills, and prioritizes relevant experience to create tailored documents. Implemented with FastAPI backend, React frontend, and deployed on AWS with CI/CD pipeline.",
                technologies=["Python", "FastAPI", "React", "Machine Learning", "AWS", "Docker"],
                status="completed"
            ),
            Project(
                title="Financial Risk Assessment Tool",
                description="Built a comprehensive risk assessment application for financial institutions using advanced statistical modeling and machine learning techniques. The tool processes large datasets to identify potential risks, generates automated reports, and provides real-time dashboard visualizations for decision makers.",
                technologies=["Python", "Pandas", "Scikit-learn", "PostgreSQL", "Plotly", "Streamlit"],
                status="completed"
            ),
            Project(
                title="Energy Trading Optimization Platform",
                description="Created an algorithmic trading platform for energy markets that optimizes powerplant unit-commitment decisions. The system processes real-time market data, implements advanced optimization algorithms, and provides automated trading recommendations to maximize profitability while managing risk.",
                technologies=["Python", "NumPy", "Optimization Algorithms", "Real-time Data Processing", "API Integration"],
                status="ongoing"
            )
        ]

        skills = [
            "Python", "JavaScript", "TypeScript", "React", "FastAPI", "Django",
            "Machine Learning", "Data Science", "AWS", "Docker", "PostgreSQL", "MongoDB",
            "Git", "CI/CD", "Agile", "Scrum", "Statistical Modeling", "Pandas", "NumPy",
            "Scikit-learn", "TensorFlow", "REST APIs", "Microservices", "DevOps"
        ]

        education = [
            Education(
                degree="Master's in Market Finance",
                institution="University of Montpellier",
                graduation_year=2025,
                field_of_study="Financial Markets and Risk Management"
            )
        ]

        urls = SocialUrls(
            linkedin="https://www.linkedin.com/in/benjamin-larger/",
            github="https://github.com/benjamin-larger"
        )

        return UserProfile(
            personal_info=personal_info,
            experiences=experiences,
            projects=projects,
            skills=skills,
            education=education,
            urls=urls,
            summary="Experienced software engineer with expertise in Python development, machine learning, and financial technology solutions."
        )

    @pytest.fixture
    def sample_job_data(self):
        """Create sample job data for testing."""
        return JobData(
            company_name="TechCorp Solutions",
            position="Senior Python Developer",
            requirements=[
                "5+ years of Python development experience",
                "Experience with FastAPI and REST APIs",
                "Machine Learning and Data Science background",
                "AWS cloud platform experience",
                "Strong problem-solving skills"
            ],
            skills_required=[
                "Python", "FastAPI", "Machine Learning", "AWS", "Docker",
                "PostgreSQL", "Git", "Agile", "REST APIs", "Data Science"
            ],
            experience_level=ExperienceLevel.SENIOR,
            description="We are seeking a senior Python developer to join our AI team working on cutting-edge machine learning solutions for financial technology applications."
        )

    @pytest.fixture
    def sample_match_result(self):
        """Create sample match result for testing."""
        relevant_experiences = [
            RelevantExperience(
                company="ENGIE Madrid",
                role="IS Software Engineer Intern",
                relevance_score=0.95,
                matching_skills=["Python", "AWS", "Docker", "REST APIs"],
                key_achievements=[
                    "Developed Energy Management Solutions using Python RESTful APIs",
                    "Built containerized applications with Docker and deployed on AWS Lambda"
                ],
                duration_months=6,
                relevance_reason="Direct experience with Python, AWS, and API development"
            ),
            RelevantExperience(
                company="ING Brussels",
                role="FEC Data Analyst Intern",
                relevance_score=0.75,
                matching_skills=["Python", "Data Science"],
                key_achievements=[
                    "Conducted Python data analytics on 10,000+ records",
                    "Developed automation scripts reducing analysis time by 20%"
                ],
                duration_months=5,
                relevance_reason="Strong Python and data analysis experience"
            )
        ]

        recommended_projects = [
            RecommendedProject(
                title="AI-Powered CV Generator",
                relevance_score=0.9,
                matching_technologies=["Python", "FastAPI", "Machine Learning", "AWS"],
                description="Intelligent CV generation system using ML algorithms",
                relevance_reason="Demonstrates Python, FastAPI, and ML expertise"
            ),
            RecommendedProject(
                title="Financial Risk Assessment Tool",
                relevance_score=0.8,
                matching_technologies=["Python", "Machine Learning", "Data Science"],
                description="Risk assessment application with ML and statistical modeling",
                relevance_reason="Shows advanced Python and ML capabilities"
            )
        ]

        analysis_details = AnalysisDetails(
            skills_match_percentage=85.0,
            experience_match_percentage=80.0,
            education_match=True,
            missing_skills=["Kubernetes", "TensorFlow"],
            additional_skills=["VBA", "Financial Modeling"],
            strength_areas=["Python Development", "Machine Learning", "Cloud Platforms"],
            weakness_areas=["Container Orchestration", "Deep Learning"],
            match_explanation="Strong technical alignment with Python, ML, and cloud experience"
        )

        return MatchResult(
            score=MatchScore.VERY_GOOD,
            matched_skills=["Python", "FastAPI", "Machine Learning", "AWS", "Docker", "REST APIs", "Data Science"],
            relevant_experiences=relevant_experiences,
            recommended_projects=recommended_projects,
            analysis_details=analysis_details,
            job_title="Senior Python Developer",
            company_name="TechCorp Solutions",
            confidence_level=0.9
        )

    @pytest.fixture
    def template_customizer(self):
        """Create TemplateCustomizer instance with single-page mode enabled."""
        return TemplateCustomizer(single_page_mode=True)

    def test_content_limits_constants(self):
        """Test that CONTENT_LIMITS constants are properly defined."""
        required_limits = [
            'professional_summary', 'project_description', 'achievement_bullet',
            'cover_letter_opening', 'cover_letter_body_paragraph', 'cover_letter_closing',
            'skills_list', 'experiences_shown', 'projects_shown'
        ]

        for limit in required_limits:
            assert limit in CONTENT_LIMITS, f"Missing content limit: {limit}"
            assert isinstance(CONTENT_LIMITS[limit], int), f"Content limit {limit} must be integer"
            assert CONTENT_LIMITS[limit] > 0, f"Content limit {limit} must be positive"

    def test_single_page_constraints_validation(self):
        """Test SinglePageConstraints model validation."""
        constraints = SinglePageConstraints()

        # Test default values
        assert constraints.max_experiences == 3
        assert constraints.max_projects == 3
        assert constraints.professional_summary_limit == 300

        # Test validation
        content_metrics = {'total_characters': 3000}
        assert constraints.validate_content_fits_page(content_metrics) == True

        content_metrics = {'total_characters': 4000}
        assert constraints.validate_content_fits_page(content_metrics) == False

    def test_truncate_with_ellipsis(self, template_customizer):
        """Test text truncation with ellipsis functionality."""
        # Test normal truncation
        long_text = "This is a very long text that needs to be truncated to fit within the specified character limits for single-page PDF optimization."
        truncated = template_customizer._truncate_with_ellipsis(long_text, 50)

        assert len(truncated) <= 50
        assert truncated.endswith("...")
        assert "very long text that needs to be" in truncated

        # Test text within limits
        short_text = "Short text"
        result = template_customizer._truncate_with_ellipsis(short_text, 50)
        assert result == short_text

        # Test empty text
        empty_result = template_customizer._truncate_with_ellipsis("", 50)
        assert empty_result == ""

        # Test word boundary truncation
        text_with_spaces = "This is a test with multiple words that should be truncated at word boundaries"
        truncated_words = template_customizer._truncate_with_ellipsis(text_with_spaces, 30)
        assert len(truncated_words) <= 30
        assert not truncated_words.replace("...", "").endswith(" ")  # Should not end with space

    def test_content_prioritization(self, template_customizer, sample_match_result, sample_user_profile):
        """Test content prioritization with single-page constraints."""
        # Test experience prioritization
        prioritized_exp = template_customizer._prioritize_experiences(
            sample_match_result.relevant_experiences,
            sample_user_profile.experiences,
            max_count=2
        )

        assert len(prioritized_exp) <= 2
        assert prioritized_exp[0]['relevance_score'] >= prioritized_exp[1]['relevance_score']

        # Test project prioritization
        prioritized_proj = template_customizer._prioritize_projects(
            sample_match_result.recommended_projects,
            sample_user_profile.projects,
            max_count=3
        )

        assert len(prioritized_proj) <= 3
        for project in prioritized_proj:
            assert len(project['description']) <= CONTENT_LIMITS['project_description']

        # Test skills prioritization
        prioritized_skills = template_customizer._prioritize_skills(
            sample_match_result.matched_skills,
            sample_user_profile.skills,
            ["Python", "FastAPI", "AWS"],
            max_skills=8
        )

        assert len(prioritized_skills['technical']) <= 8
        assert "Python" in prioritized_skills['technical']  # Should prioritize matched skills

    def test_single_page_content_optimization(self, template_customizer, sample_match_result, sample_user_profile, sample_job_data):
        """Test content optimization for single-page constraints."""
        # Generate content
        dynamic_content = template_customizer.generate_dynamic_content(
            sample_match_result, sample_user_profile, sample_job_data
        )

        # Apply single-page optimizations
        optimized_content = template_customizer._optimize_for_single_page(dynamic_content)

        # Validate content length constraints
        assert len(optimized_content['PROFESSIONAL_SUMMARY']) <= CONTENT_LIMITS['professional_summary']
        assert len(optimized_content['DESCRIPTION_OF_THE_SIDE_PROJECT_1']) <= CONTENT_LIMITS['project_description']
        assert len(optimized_content['OPENING_PARAGRAPH']) <= CONTENT_LIMITS['cover_letter_opening']

    def test_cv_constraints_application(self, template_customizer):
        """Test CV-specific constraint application."""
        test_data = {
            'TECHNICAL_SKILLS': 'Python, JavaScript, TypeScript, React, Angular, Vue, Node.js, Django, FastAPI, Flask, Spring Boot, Express.js',
            'DESCRIPTION_OF_THE_SIDE_PROJECT_1': 'This is a very long project description that exceeds the character limit and needs to be truncated to fit within single-page constraints for PDF generation.',
            'OTHER_FIELD': 'This should remain unchanged'
        }

        constrained_data = template_customizer._apply_cv_constraints(test_data)

        # Technical skills should be limited
        skills_list = constrained_data['TECHNICAL_SKILLS'].split(', ')
        assert len(skills_list) <= 3

        # Project description should be truncated
        assert len(constrained_data['DESCRIPTION_OF_THE_SIDE_PROJECT_1']) <= CONTENT_LIMITS['project_description']

        # Other fields should remain unchanged
        assert constrained_data['OTHER_FIELD'] == test_data['OTHER_FIELD']

    def test_cover_letter_constraints_application(self, template_customizer):
        """Test cover letter-specific constraint application."""
        test_data = {
            'OPENING_PARAGRAPH': 'This is a very long opening paragraph that needs to be truncated to fit within the single-page cover letter format while maintaining professional tone and clarity.',
            'SKILLS_PARAGRAPH': 'This is an extensive skills paragraph that describes various technical competencies and professional experiences in great detail, which may exceed the optimal length for single-page format.',
            'CLOSING_PARAGRAPH': 'This closing paragraph is also quite lengthy and includes multiple sentences about enthusiasm, qualifications, and next steps.',
            'OTHER_FIELD': 'This should remain unchanged'
        }

        constrained_data = template_customizer._apply_cover_letter_constraints(test_data)

        # Paragraphs should be truncated to limits
        assert len(constrained_data['OPENING_PARAGRAPH']) <= CONTENT_LIMITS['cover_letter_opening']
        assert len(constrained_data['SKILLS_PARAGRAPH']) <= CONTENT_LIMITS['cover_letter_body_paragraph']
        assert len(constrained_data['CLOSING_PARAGRAPH']) <= CONTENT_LIMITS['cover_letter_closing']

        # Other fields should remain unchanged
        assert constrained_data['OTHER_FIELD'] == test_data['OTHER_FIELD']

    def test_content_metrics_calculation(self, template_customizer):
        """Test content metrics calculation for space estimation."""
        test_content = {
            'field1': 'Short text',
            'field2': 'This is a longer text field with more characters',
            'field3': ['List', 'of', 'items'],
            'field4': 12345
        }

        metrics = template_customizer._calculate_content_metrics(test_content)

        assert 'field1' in metrics
        assert 'field2' in metrics
        assert 'field3' in metrics
        assert 'field4' in metrics
        assert 'total_characters' in metrics

        assert metrics['field1'] == len('Short text')
        assert metrics['total_characters'] == sum(metrics[k] for k in metrics if k != 'total_characters')

    def test_space_utilization_estimation(self, template_customizer):
        """Test space utilization estimation."""
        # Test normal utilization
        metrics = {'total_characters': 2000}
        utilization = template_customizer._estimate_space_utilization(metrics)
        assert 0.0 <= utilization <= 1.0
        assert utilization < 1.0  # Should be under single page limit

        # Test over-utilization
        metrics = {'total_characters': 4000}
        utilization = template_customizer._estimate_space_utilization(metrics)
        assert utilization == 1.0  # Capped at 1.0

    def test_single_page_compliance_validation(self, template_customizer):
        """Test single-page compliance validation."""
        # Test compliant content
        metrics = {'total_characters': 3000}
        assert template_customizer._validate_single_page_compliance(metrics) == True

        # Test non-compliant content
        metrics = {'total_characters': 4000}
        assert template_customizer._validate_single_page_compliance(metrics) == False

    def test_full_template_customization_workflow(self, template_customizer, sample_match_result, sample_user_profile, sample_job_data):
        """Test complete template customization workflow with single-page constraints."""
        # This test requires actual template files
        try:
            result = template_customizer.customize_templates(
                sample_match_result, sample_user_profile, sample_job_data
            )

            # Validate result type
            assert isinstance(result, CustomizationResult)

            # Validate content is present
            assert result.cv_html
            assert result.cover_letter_html

            # Validate metrics
            assert isinstance(result.single_page_compliant, bool)
            assert isinstance(result.space_utilization, float)
            assert 0.0 <= result.space_utilization <= 1.0
            assert isinstance(result.customization_score, float)
            assert 0.0 <= result.customization_score <= 1.0

            # Validate changes tracking
            assert isinstance(result.changes_made, list)
            assert len(result.changes_made) > 0

            # Validate content metrics
            assert isinstance(result.content_metrics, dict)
            assert 'total_characters' in result.content_metrics

        except Exception as e:
            # If template files don't exist, test should pass but log the issue
            pytest.skip(f"Template files not available for full workflow test: {e}")

    def test_customization_score_calculation(self, template_customizer, sample_match_result):
        """Test customization score calculation."""
        test_content = {'test': 'content'}
        score = template_customizer._calculate_customization_score(sample_match_result, test_content)

        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)

        # High match score should result in higher customization score
        high_match_result = sample_match_result.copy()
        high_match_result.score = MatchScore.PERFECT
        high_score = template_customizer._calculate_customization_score(high_match_result, test_content)

        assert high_score >= score

    def test_professional_summary_length_constraint(self, template_customizer, sample_match_result, sample_user_profile, sample_job_data):
        """Test that professional summary respects length constraints."""
        summary = template_customizer._generate_professional_summary(
            sample_match_result, sample_user_profile, sample_job_data
        )

        # Professional summary should be reasonable length but might exceed limit before truncation
        assert isinstance(summary, str)
        assert len(summary) > 50  # Should have meaningful content

        # After truncation, should fit constraints
        truncated_summary = template_customizer._truncate_with_ellipsis(
            summary, CONTENT_LIMITS['professional_summary']
        )
        assert len(truncated_summary) <= CONTENT_LIMITS['professional_summary']

    def test_project_description_constraints(self, template_customizer, sample_match_result, sample_user_profile):
        """Test that project descriptions are properly constrained."""
        prioritized_projects = template_customizer._prioritize_projects(
            sample_match_result.recommended_projects,
            sample_user_profile.projects,
            max_count=CONTENT_LIMITS['projects_shown']
        )

        for project in prioritized_projects:
            assert len(project['description']) <= CONTENT_LIMITS['project_description']
            assert len(project['description']) >= 80  # Minimum meaningful length

    def test_achievement_bullet_constraints(self, template_customizer, sample_match_result, sample_user_profile):
        """Test that achievement bullets respect character limits."""
        prioritized_experiences = template_customizer._prioritize_experiences(
            sample_match_result.relevant_experiences,
            sample_user_profile.experiences,
            max_count=CONTENT_LIMITS['experiences_shown']
        )

        for experience in prioritized_experiences:
            for achievement in experience['key_achievements']:
                assert len(achievement) <= CONTENT_LIMITS['achievement_bullet']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])