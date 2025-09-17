"""
Single-Page Template Customizer Usage Example

This example demonstrates the complete workflow of using the enhanced TemplateCustomizer
with strict single-page PDF constraints. It shows how to generate optimized templates
that fit within single-page limitations while maintaining professional quality.
"""

import sys
from pathlib import Path
from datetime import datetime, date

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.template_customizer import TemplateCustomizer, CONTENT_LIMITS
from src.models.match_result import (
    MatchResult, RelevantExperience, RecommendedProject,
    AnalysisDetails, MatchScore
)
from src.models.user_profile import (
    UserProfile, PersonalInfo, WorkExperience, Project,
    Education, SocialUrls
)
from src.models.job_data import JobData, ExperienceLevel


def create_sample_data():
    """Create comprehensive sample data for demonstration."""

    # Create user profile
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
                "Developed Energy Management Solutions using Python RESTful APIs for powerplant optimization algorithms and real-time market integration",
                "Implemented comprehensive web scraping solutions with Playwright to process business-critical information from multiple internal systems",
                "Built and deployed containerized applications with Docker on AWS Lambda infrastructure for improved scalability and cost efficiency",
                "Collaborated extensively with Back and Middle Office teams to implement tailored software solutions for energy trading operations",
                "Optimized existing codebase performance resulting in 40% faster execution times and reduced memory consumption"
            ],
            technologies=["Python", "Docker", "AWS Lambda", "Playwright", "RESTful APIs", "Energy Trading"]
        ),
        WorkExperience(
            company="ING Brussels",
            role="FEC Data Analyst Intern",
            start_date=date(2023, 5, 1),
            end_date=date(2023, 9, 30),
            achievements=[
                "Conducted comprehensive Python data analytics to identify suspicious behaviors and patterns in financial datasets containing 10,000+ transaction records",
                "Developed advanced VBA automation scripts that reduced manual analysis time by 20% while improving accuracy and consistency of reporting",
                "Created detailed compliance reports and interactive visualizations for senior management using advanced statistical methods and data visualization tools",
                "Implemented machine learning algorithms for anomaly detection in financial transactions improving detection rates by 15%"
            ],
            technologies=["Python", "VBA", "Excel", "Data Analysis", "Statistical Modeling", "Machine Learning"]
        ),
        WorkExperience(
            company="TechStart Innovation Lab",
            role="Junior Full Stack Developer",
            start_date=date(2022, 9, 1),
            end_date=date(2022, 12, 31),
            achievements=[
                "Built responsive web applications using React and Node.js for startup clients in fintech and e-commerce sectors",
                "Implemented secure authentication systems and user management features using JWT and OAuth2 protocols",
                "Developed RESTful APIs with comprehensive documentation and automated testing suites for improved reliability"
            ],
            technologies=["React", "Node.js", "JavaScript", "MongoDB", "JWT", "OAuth2"]
        )
    ]

    projects = [
        Project(
            title="AI-Powered Intelligent CV Generator System",
            description="Developed a sophisticated AI-powered CV generation system using Python, FastAPI, and advanced machine learning algorithms that automatically customizes resumes based on job requirements and candidate profiles. The system analyzes job descriptions using natural language processing, matches user skills and experience with job requirements, and prioritizes relevant content to create highly targeted professional documents. Implemented with modern tech stack including FastAPI backend, React frontend with TypeScript, PostgreSQL database, and deployed on AWS cloud infrastructure with comprehensive CI/CD pipeline using GitHub Actions and Docker containers.",
            technologies=["Python", "FastAPI", "React", "TypeScript", "Machine Learning", "NLP", "AWS", "Docker", "PostgreSQL", "GitHub Actions"],
            status="completed"
        ),
        Project(
            title="Advanced Financial Risk Assessment and Analytics Platform",
            description="Built a comprehensive enterprise-grade risk assessment application for financial institutions using advanced statistical modeling, machine learning techniques, and real-time data processing capabilities. The platform processes large-scale financial datasets to identify potential risks, generates automated compliance reports, provides real-time dashboard visualizations for decision makers, and implements predictive analytics for risk forecasting. Features include Monte Carlo simulations, stress testing modules, regulatory compliance tracking, and integration with external market data providers through secure APIs.",
            technologies=["Python", "Pandas", "Scikit-learn", "TensorFlow", "PostgreSQL", "Redis", "Plotly", "Streamlit", "Apache Kafka", "REST APIs"],
            status="completed"
        ),
        Project(
            title="Energy Trading Optimization and Market Analysis Platform",
            description="Created a sophisticated algorithmic trading platform for energy markets that optimizes powerplant unit-commitment decisions using advanced mathematical optimization algorithms and machine learning models. The system processes real-time market data from multiple sources, implements complex optimization algorithms for profit maximization, provides automated trading recommendations, manages risk exposure, and includes comprehensive backtesting capabilities. Built with microservices architecture for scalability and reliability.",
            technologies=["Python", "NumPy", "SciPy", "Optimization Algorithms", "Real-time Data Processing", "Apache Spark", "Microservices", "API Integration"],
            status="ongoing"
        ),
        Project(
            title="Blockchain-Based Supply Chain Transparency Solution",
            description="Developed a blockchain-based application for supply chain transparency and traceability using Ethereum smart contracts and Web3 technologies. The solution enables end-to-end tracking of products from manufacturing to consumer delivery, implements automated compliance verification, and provides immutable audit trails for regulatory purposes.",
            technologies=["Solidity", "Ethereum", "Web3.js", "React", "Node.js", "IPFS", "Smart Contracts"],
            status="completed"
        )
    ]

    skills = [
        "Python", "JavaScript", "TypeScript", "React", "FastAPI", "Django", "Flask",
        "Machine Learning", "Data Science", "Natural Language Processing", "TensorFlow", "Scikit-learn",
        "AWS", "Docker", "Kubernetes", "PostgreSQL", "MongoDB", "Redis", "Apache Kafka",
        "Git", "GitHub Actions", "CI/CD", "Agile", "Scrum", "DevOps", "Microservices",
        "Statistical Modeling", "Pandas", "NumPy", "Plotly", "Streamlit", "Jupyter",
        "REST APIs", "GraphQL", "OAuth2", "JWT", "Security", "Blockchain", "Solidity",
        "Linux", "Bash", "SQL", "NoSQL", "Elasticsearch", "Monitoring", "Testing"
    ]

    education = [
        Education(
            degree="Master's in Market Finance",
            institution="University of Montpellier",
            graduation_year=2025,
            field_of_study="Financial Markets and Risk Management"
        ),
        Education(
            degree="42 School Programming Engineering",
            institution="42 Málaga Fundación Telefónica",
            graduation_year=2025,
            field_of_study="Computer Science and Software Engineering"
        )
    ]

    urls = SocialUrls(
        linkedin="https://www.linkedin.com/in/benjamin-larger/",
        github="https://github.com/benjamin-larger",
        portfolio="https://benjamin-larger.dev"
    )

    user_profile = UserProfile(
        personal_info=personal_info,
        experiences=experiences,
        projects=projects,
        skills=skills,
        education=education,
        urls=urls,
        summary="Experienced software engineer with expertise in Python development, machine learning, financial technology solutions, and full-stack web development. Proven track record in delivering scalable applications and data-driven solutions."
    )

    # Create job data
    job_data = JobData(
        company_name="TechCorp Solutions",
        position="Senior Python Developer - AI/ML Team",
        requirements=[
            "5+ years of Python development experience with focus on backend systems",
            "Strong experience with FastAPI, Django, or similar Python web frameworks",
            "Machine Learning and Data Science background with hands-on ML model development",
            "AWS cloud platform experience including Lambda, EC2, RDS, and S3 services",
            "Experience with containerization using Docker and orchestration with Kubernetes",
            "Strong problem-solving skills and ability to work in agile development environment",
            "Experience with database design and optimization (PostgreSQL, MongoDB)",
            "Knowledge of software engineering best practices including testing and CI/CD"
        ],
        skills_required=[
            "Python", "FastAPI", "Django", "Machine Learning", "TensorFlow", "Scikit-learn",
            "AWS", "Docker", "Kubernetes", "PostgreSQL", "MongoDB", "Git", "CI/CD",
            "Agile", "REST APIs", "Data Science", "Pandas", "NumPy", "Microservices"
        ],
        experience_level=ExperienceLevel.SENIOR,
        description="We are seeking a senior Python developer to join our AI/ML team working on cutting-edge machine learning solutions for financial technology applications. You will be responsible for designing and implementing scalable backend systems, developing ML models, and collaborating with cross-functional teams to deliver innovative solutions."
    )

    # Create match result
    relevant_experiences = [
        RelevantExperience(
            company="ENGIE Madrid",
            role="IS Software Engineer Intern",
            relevance_score=0.95,
            matching_skills=["Python", "AWS", "Docker", "REST APIs", "Energy Trading"],
            key_achievements=[
                "Developed Energy Management Solutions using Python RESTful APIs for powerplant optimization",
                "Built containerized applications with Docker and deployed on AWS Lambda for scalability",
                "Collaborated with teams to implement tailored software solutions for energy trading"
            ],
            duration_months=6,
            relevance_reason="Direct experience with Python, AWS, Docker, and API development aligns perfectly with job requirements"
        ),
        RelevantExperience(
            company="ING Brussels",
            role="FEC Data Analyst Intern",
            relevance_score=0.85,
            matching_skills=["Python", "Data Science", "Machine Learning", "Statistical Modeling"],
            key_achievements=[
                "Conducted Python data analytics on 10,000+ financial transaction records",
                "Implemented machine learning algorithms for anomaly detection improving detection rates",
                "Developed automation scripts reducing analysis time while improving accuracy"
            ],
            duration_months=5,
            relevance_reason="Strong Python, data science, and machine learning experience directly applicable to AI/ML team role"
        )
    ]

    recommended_projects = [
        RecommendedProject(
            title="AI-Powered Intelligent CV Generator System",
            relevance_score=0.95,
            matching_technologies=["Python", "FastAPI", "Machine Learning", "AWS", "Docker", "PostgreSQL"],
            description="AI-powered system using Python, FastAPI, and ML algorithms for intelligent document generation",
            relevance_reason="Demonstrates advanced Python, FastAPI, ML, and AWS expertise directly relevant to the role"
        ),
        RecommendedProject(
            title="Advanced Financial Risk Assessment Platform",
            relevance_score=0.90,
            matching_technologies=["Python", "Machine Learning", "TensorFlow", "Scikit-learn", "Data Science"],
            description="Enterprise risk assessment platform with ML models and real-time analytics",
            relevance_reason="Shows expertise in Python ML ecosystem and enterprise-scale application development"
        ),
        RecommendedProject(
            title="Energy Trading Optimization Platform",
            relevance_score=0.80,
            matching_technologies=["Python", "Optimization Algorithms", "Microservices", "Real-time Processing"],
            description="Algorithmic trading platform with optimization algorithms and microservices architecture",
            relevance_reason="Demonstrates advanced Python development and system architecture skills"
        )
    ]

    analysis_details = AnalysisDetails(
        skills_match_percentage=92.0,
        experience_match_percentage=88.0,
        education_match=True,
        missing_skills=["Kubernetes", "GraphQL"],
        additional_skills=["Blockchain", "Solidity", "Energy Trading", "VBA"],
        strength_areas=["Python Development", "Machine Learning", "Cloud Platforms", "Data Science", "API Development"],
        weakness_areas=["Container Orchestration", "GraphQL APIs"],
        match_explanation="Exceptional technical alignment with strong Python, ML, and cloud experience. Minor gaps in Kubernetes and GraphQL easily addressable."
    )

    match_result = MatchResult(
        score=MatchScore.EXCELLENT,
        matched_skills=["Python", "FastAPI", "Machine Learning", "TensorFlow", "Scikit-learn", "AWS", "Docker", "PostgreSQL", "REST APIs", "Data Science", "Pandas", "NumPy"],
        relevant_experiences=relevant_experiences,
        recommended_projects=recommended_projects,
        analysis_details=analysis_details,
        job_title="Senior Python Developer - AI/ML Team",
        company_name="TechCorp Solutions",
        confidence_level=0.95
    )

    return user_profile, job_data, match_result


def demonstrate_single_page_constraints():
    """Demonstrate single-page constraint enforcement."""
    print("=" * 80)
    print("SINGLE-PAGE PDF CONSTRAINT DEMONSTRATION")
    print("=" * 80)

    print(f"\nContent Limits Configuration:")
    for key, value in CONTENT_LIMITS.items():
        print(f"  {key}: {value}")

    # Create template customizer with single-page mode
    customizer = TemplateCustomizer(single_page_mode=True)
    print(f"\nTemplateCustomizer initialized with single_page_mode=True")
    print(f"Constraints: max_experiences={customizer.constraints.max_experiences}, max_projects={customizer.constraints.max_projects}")


def demonstrate_content_optimization():
    """Demonstrate content optimization for single-page constraints."""
    print("\n" + "=" * 80)
    print("CONTENT OPTIMIZATION DEMONSTRATION")
    print("=" * 80)

    customizer = TemplateCustomizer(single_page_mode=True)

    # Test text truncation
    long_text = ("This is a very long project description that demonstrates how the system handles "
                "content that exceeds the character limits imposed by single-page PDF constraints. "
                "The truncation algorithm will intelligently shorten this text while preserving "
                "meaning and adding ellipsis to indicate truncation. This ensures that all content "
                "fits within the specified page limits while maintaining readability and professional appearance.")

    truncated = customizer._truncate_with_ellipsis(long_text, CONTENT_LIMITS['project_description'])

    print(f"\nOriginal text length: {len(long_text)} characters")
    print(f"Truncated text length: {len(truncated)} characters")
    print(f"Limit: {CONTENT_LIMITS['project_description']} characters")
    print(f"Original: {long_text}")
    print(f"Truncated: {truncated}")

    # Test content metrics
    sample_content = {
        'PROFESSIONAL_SUMMARY': 'Experienced software engineer with expertise in Python development...',
        'TECHNICAL_SKILLS': 'Python, FastAPI, Machine Learning, AWS, Docker',
        'PROJECT_1_DESC': 'AI-powered system for intelligent document generation',
        'PROJECT_2_DESC': 'Financial risk assessment platform with ML models'
    }

    metrics = customizer._calculate_content_metrics(sample_content)
    utilization = customizer._estimate_space_utilization(metrics)
    compliant = customizer._validate_single_page_compliance(metrics)

    print(f"\nContent Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value} characters")

    print(f"\nSpace Utilization: {utilization:.3f} ({utilization*100:.1f}%)")
    print(f"Single Page Compliant: {compliant}")


def demonstrate_full_workflow():
    """Demonstrate the complete template customization workflow."""
    print("\n" + "=" * 80)
    print("COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 80)

    # Create sample data
    user_profile, job_data, match_result = create_sample_data()

    print(f"\nInput Data Summary:")
    print(f"  User: {user_profile.personal_info.name}")
    print(f"  Job: {job_data.position} at {job_data.company_name}")
    print(f"  Match Score: {match_result.score}/10")
    print(f"  Matched Skills: {len(match_result.matched_skills)}")
    print(f"  Relevant Experiences: {len(match_result.relevant_experiences)}")
    print(f"  Recommended Projects: {len(match_result.recommended_projects)}")

    # Create customizer and generate content
    customizer = TemplateCustomizer(single_page_mode=True)

    try:
        # Generate dynamic content
        dynamic_content = customizer.generate_dynamic_content(match_result, user_profile, job_data)

        print(f"\nGenerated Content Summary:")
        print(f"  Professional Summary: {len(dynamic_content['PROFESSIONAL_SUMMARY'])} chars")
        print(f"  Project 1 Description: {len(dynamic_content['DESCRIPTION_OF_THE_SIDE_PROJECT_1'])} chars")
        print(f"  Project 2 Description: {len(dynamic_content['DESCRIPTION_OF_THE_SIDE_PROJECT_2'])} chars")
        print(f"  Opening Paragraph: {len(dynamic_content['OPENING_PARAGRAPH'])} chars")

        # Apply optimizations
        optimized_content = customizer._optimize_for_single_page(dynamic_content)

        print(f"\nOptimized Content Summary:")
        print(f"  Professional Summary: {len(optimized_content['PROFESSIONAL_SUMMARY'])} chars (limit: {CONTENT_LIMITS['professional_summary']})")
        print(f"  Project 1 Description: {len(optimized_content['DESCRIPTION_OF_THE_SIDE_PROJECT_1'])} chars (limit: {CONTENT_LIMITS['project_description']})")
        print(f"  Project 2 Description: {len(optimized_content['DESCRIPTION_OF_THE_SIDE_PROJECT_2'])} chars (limit: {CONTENT_LIMITS['project_description']})")
        print(f"  Opening Paragraph: {len(optimized_content['OPENING_PARAGRAPH'])} chars (limit: {CONTENT_LIMITS['cover_letter_opening']})")

        # Calculate metrics
        metrics = customizer._calculate_content_metrics(optimized_content)
        utilization = customizer._estimate_space_utilization(metrics)
        compliant = customizer._validate_single_page_compliance(metrics)

        print(f"\nOptimization Results:")
        print(f"  Total Content: {metrics['total_characters']} characters")
        print(f"  Space Utilization: {utilization:.3f} ({utilization*100:.1f}%)")
        print(f"  Single Page Compliant: {compliant}")

        # Show sample content
        print(f"\nSample Optimized Content:")
        print(f"  Professional Summary: {optimized_content['PROFESSIONAL_SUMMARY']}")
        print(f"  Project 1: {optimized_content['DESCRIPTION_OF_THE_SIDE_PROJECT_1']}")

        # Attempt full template customization if templates exist
        try:
            result = customizer.customize_templates(match_result, user_profile, job_data)

            print(f"\nTemplate Customization Results:")
            print(f"  CV HTML Generated: {len(result.cv_html)} characters")
            print(f"  Cover Letter HTML Generated: {len(result.cover_letter_html)} characters")
            print(f"  Customization Score: {result.customization_score:.3f}")
            print(f"  Single Page Compliant: {result.single_page_compliant}")
            print(f"  Space Utilization: {result.space_utilization:.3f}")
            print(f"  Changes Made: {len(result.changes_made)}")

            print(f"\nChanges Applied:")
            for change in result.changes_made[:5]:  # Show first 5 changes
                print(f"  - {change}")
            if len(result.changes_made) > 5:
                print(f"  ... and {len(result.changes_made) - 5} more changes")

        except Exception as e:
            print(f"\nTemplate files not available for full customization: {e}")
            print("This is expected if running outside the full project environment.")

    except Exception as e:
        print(f"\nError in workflow demonstration: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_constraint_validation():
    """Demonstrate constraint validation and compliance checking."""
    print("\n" + "=" * 80)
    print("CONSTRAINT VALIDATION DEMONSTRATION")
    print("=" * 80)

    customizer = TemplateCustomizer(single_page_mode=True)

    # Test various content scenarios
    scenarios = [
        {"name": "Compliant Content", "total_characters": 3000},
        {"name": "At Limit Content", "total_characters": 3500},
        {"name": "Over Limit Content", "total_characters": 4000},
        {"name": "Significantly Over Limit", "total_characters": 5000}
    ]

    print(f"\nConstraint Validation Results:")
    print(f"{'Scenario':<25} {'Characters':<12} {'Utilization':<12} {'Compliant':<10}")
    print("-" * 65)

    for scenario in scenarios:
        metrics = {"total_characters": scenario["total_characters"]}
        utilization = customizer._estimate_space_utilization(metrics)
        compliant = customizer._validate_single_page_compliance(metrics)

        print(f"{scenario['name']:<25} {scenario['total_characters']:<12} {utilization:.3f}       {compliant}")


def main():
    """Main demonstration function."""
    print("Single-Page PDF Template Customizer Demonstration")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    demonstrate_single_page_constraints()
    demonstrate_content_optimization()
    demonstrate_constraint_validation()
    demonstrate_full_workflow()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("✓ Content length limits for single-page PDF optimization")
    print("✓ Intelligent text truncation with word boundary preservation")
    print("✓ Space utilization estimation and compliance validation")
    print("✓ Content prioritization based on job matching relevance")
    print("✓ Comprehensive metrics tracking and reporting")
    print("✓ Full workflow integration with error handling")

    print(f"\nThe TemplateCustomizer ensures all generated content fits within single-page")
    print(f"PDF constraints while maintaining professional quality and relevance.")


if __name__ == "__main__":
    main()