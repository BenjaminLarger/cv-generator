"""
ProfileMatcher Example Usage

This example demonstrates how to use the ProfileMatcher class to analyze
compatibility between user profiles and job postings. It shows real-world
usage patterns and best practices for profile-job matching analysis.
"""

import os
from datetime import date
from pathlib import Path

# Add the src directory to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.profile_matcher import ProfileMatcher, ProfileMatchingError
from src.models.job_data import JobData, ExperienceLevel
from src.models.user_profile import (
    UserProfile, PersonalInfo, WorkExperience, Project, Education, SocialUrls
)
from src.models.match_result import MatchScore


def create_sample_job_data() -> JobData:
    """Create a sample job posting for demonstration."""
    return JobData(
        company_name="InnovateTech Solutions",
        position="Senior Full Stack Developer",
        skills_required=[
            "Python", "Django", "React", "JavaScript", "PostgreSQL",
            "AWS", "Docker", "Git", "REST APIs", "Agile Development"
        ],
        requirements=[
            "5+ years of software development experience",
            "Bachelor's degree in Computer Science or related field",
            "Strong experience with Python web frameworks",
            "Frontend development experience with modern JavaScript frameworks",
            "Experience with cloud platforms (AWS preferred)",
            "Knowledge of containerization technologies",
            "Strong problem-solving and communication skills",
            "Experience with Agile/Scrum methodologies"
        ],
        experience_level=ExperienceLevel.SENIOR,
        description="""
        We are looking for a Senior Full Stack Developer to join our growing engineering team.
        You will be responsible for developing and maintaining our core web applications,
        working on both frontend and backend components. This role offers the opportunity
        to work with cutting-edge technologies and contribute to products that impact
        thousands of users daily.

        Key Responsibilities:
        - Design and implement scalable web applications
        - Collaborate with cross-functional teams to define and implement new features
        - Write clean, maintainable, and well-tested code
        - Participate in code reviews and technical discussions
        - Optimize applications for maximum speed and scalability
        - Stay up-to-date with emerging technologies and industry trends

        We offer competitive compensation, comprehensive benefits, flexible work arrangements,
        and opportunities for professional growth in a collaborative environment.
        """,
        url="https://innovatetech.com/careers/senior-fullstack-developer"
    )


def create_sample_user_profile() -> UserProfile:
    """Create a sample user profile for demonstration."""
    personal_info = PersonalInfo(
        name="Alexandra Chen",
        email="alexandra.chen@email.com",
        phone="+1 (555) 123-4567",
        location="San Francisco, CA"
    )

    work_experiences = [
        WorkExperience(
            company="TechStartup Inc",
            role="Full Stack Developer",
            start_date=date(2021, 3, 1),
            end_date=None,  # Current position
            achievements=[
                "Developed and deployed 5+ web applications serving 10,000+ daily active users",
                "Improved application performance by 40% through database optimization and caching",
                "Led migration from monolithic to microservices architecture",
                "Mentored 2 junior developers and conducted code reviews",
                "Implemented automated testing pipeline reducing bugs by 60%"
            ],
            technologies=[
                "Python", "Django", "React", "JavaScript", "PostgreSQL",
                "Redis", "Docker", "AWS EC2", "Git", "Jenkins"
            ],
            location="San Francisco, CA"
        ),
        WorkExperience(
            company="WebSolutions LLC",
            role="Python Developer",
            start_date=date(2019, 6, 1),
            end_date=date(2021, 2, 28),
            achievements=[
                "Built RESTful APIs handling 1M+ requests per day",
                "Developed automated data processing pipelines",
                "Integrated third-party payment systems (Stripe, PayPal)",
                "Reduced server costs by 30% through optimization",
                "Collaborated with design team on user experience improvements"
            ],
            technologies=[
                "Python", "Flask", "SQLAlchemy", "MySQL", "jQuery",
                "Bootstrap", "AWS S3", "Celery", "Nginx"
            ],
            location="Remote"
        ),
        WorkExperience(
            company="Digital Agency Pro",
            role="Junior Web Developer",
            start_date=date(2018, 1, 1),
            end_date=date(2019, 5, 31),
            achievements=[
                "Developed responsive websites for 20+ clients",
                "Maintained and updated existing web applications",
                "Learned and implemented new frontend frameworks",
                "Participated in client meetings and requirement gathering"
            ],
            technologies=[
                "HTML", "CSS", "JavaScript", "PHP", "WordPress",
                "MySQL", "Git", "Adobe Creative Suite"
            ],
            location="New York, NY"
        )
    ]

    education = [
        Education(
            degree="Bachelor of Science",
            field_of_study="Computer Science",
            institution="University of California, Berkeley",
            graduation_year=2017,
            gpa=3.7,
            honors="Magna Cum Laude",
            location="Berkeley, CA"
        )
    ]

    projects = [
        Project(
            title="E-Commerce Platform",
            description="""
            Built a comprehensive e-commerce platform from scratch using Django and React.
            Features include user authentication, product catalog, shopping cart, payment
            processing, order management, and admin dashboard. Implemented real-time
            inventory tracking and automated email notifications. The platform handles
            concurrent users and includes comprehensive testing coverage.
            """,
            technologies=[
                "Python", "Django", "Django REST Framework", "React",
                "Redux", "PostgreSQL", "Redis", "Celery", "Stripe API",
                "AWS S3", "Docker", "pytest"
            ],
            url="https://github.com/alexandra-chen/ecommerce-platform",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 1),
            status="completed"
        ),
        Project(
            title="Real-Time Chat Application",
            description="""
            Developed a real-time chat application with WebSocket support using Django
            Channels and React. Features include private messaging, group chats, file
            sharing, message history, and user presence indicators. Implemented
            comprehensive authentication and authorization system.
            """,
            technologies=[
                "Python", "Django", "Django Channels", "WebSocket",
                "React", "Socket.io", "Redis", "PostgreSQL", "JWT"
            ],
            url="https://github.com/alexandra-chen/chat-app",
            start_date=date(2022, 8, 1),
            end_date=date(2022, 12, 1),
            status="completed"
        ),
        Project(
            title="Task Management API",
            description="""
            Created a RESTful API for task management with advanced features like
            task dependencies, time tracking, project management, and team collaboration.
            Includes comprehensive documentation, automated testing, and deployment pipeline.
            """,
            technologies=[
                "Python", "FastAPI", "SQLAlchemy", "PostgreSQL",
                "Docker", "GitHub Actions", "Swagger", "pytest"
            ],
            url="https://github.com/alexandra-chen/task-api",
            start_date=date(2023, 7, 1),
            status="ongoing"
        )
    ]

    skills = [
        # Backend Technologies
        "Python", "Django", "Flask", "FastAPI", "Django REST Framework",

        # Frontend Technologies
        "React", "JavaScript", "TypeScript", "HTML5", "CSS3",
        "Redux", "jQuery", "Bootstrap", "Sass",

        # Databases
        "PostgreSQL", "MySQL", "SQLAlchemy", "Redis",

        # DevOps & Cloud
        "Docker", "AWS", "AWS EC2", "AWS S3", "Jenkins", "GitHub Actions",

        # Tools & Methodologies
        "Git", "Linux", "Nginx", "Agile", "Scrum", "TDD",
        "RESTful APIs", "WebSocket", "Microservices",

        # Soft Skills
        "Problem Solving", "Team Leadership", "Code Review",
        "Technical Mentoring", "Client Communication"
    ]

    social_urls = SocialUrls(
        github="https://github.com/alexandra-chen",
        linkedin="https://linkedin.com/in/alexandra-chen-dev",
        portfolio="https://alexandra-chen.dev"
    )

    summary = """
    Passionate Full Stack Developer with 5+ years of experience building scalable
    web applications. Expertise in Python, Django, and React with a strong focus
    on clean code, testing, and performance optimization. Proven track record of
    leading technical projects, mentoring junior developers, and delivering
    high-quality solutions that drive business growth. Passionate about learning
    new technologies and contributing to open-source projects.
    """

    return UserProfile(
        personal_info=personal_info,
        experiences=work_experiences,
        skills=skills,
        education=education,
        projects=projects,
        urls=social_urls,
        summary=summary
    )


def demonstrate_skill_matching():
    """Demonstrate skill matching analysis."""
    print("=== SKILL MATCHING ANALYSIS ===\n")

    # Initialize matcher
    matcher = ProfileMatcher()

    job_data = create_sample_job_data()
    user_profile = create_sample_user_profile()

    # Perform skill matching
    skill_match_result = matcher.calculate_skill_match(
        job_data.skills_required,
        user_profile.skills
    )

    print(f"📊 Skill Match Analysis:")
    print(f"   Overall Match: {skill_match_result['match_percentage']:.1f}%")
    print(f"   Direct Matches ({len(skill_match_result['direct_matches'])}): {', '.join(skill_match_result['direct_matches'][:8])}")

    if skill_match_result['related_matches']:
        print(f"   Related Skills ({len(skill_match_result['related_matches'])}):")
        for user_skill, job_skill, confidence in skill_match_result['related_matches'][:3]:
            print(f"     • {user_skill} → {job_skill} (confidence: {confidence:.1f})")

    if skill_match_result['missing_skills']:
        print(f"   Missing Skills ({len(skill_match_result['missing_skills'])}): {', '.join(skill_match_result['missing_skills'][:5])}")

    if skill_match_result['transferable_skills']:
        print(f"   Transferable Skills: {', '.join(skill_match_result['transferable_skills'][:5])}")

    print()


def demonstrate_experience_selection():
    """Demonstrate experience relevance analysis."""
    print("=== EXPERIENCE RELEVANCE ANALYSIS ===\n")

    matcher = ProfileMatcher()
    job_data = create_sample_job_data()
    user_profile = create_sample_user_profile()

    # Convert experiences to dict format for the method
    experiences_dict = []
    for exp in user_profile.experiences:
        experiences_dict.append({
            'company': exp.company,
            'role': exp.role,
            'technologies': exp.technologies,
            'achievements': exp.achievements,
            'start_date': exp.start_date,
            'end_date': exp.end_date
        })

    relevant_experiences = matcher.select_relevant_experiences(experiences_dict, job_data)

    print(f"📋 Experience Relevance (Top {len(relevant_experiences)}):")
    for i, exp in enumerate(relevant_experiences, 1):
        print(f"   {i}. {exp['role']} at {exp['company']}")
        print(f"      Relevance Score: {exp['relevance_score']:.2f}")
        duration = "Present" if not exp['end_date'] else exp['end_date'].strftime('%Y-%m')
        print(f"      Duration: {exp['start_date'].strftime('%Y-%m')} - {duration}")
        print(f"      Key Technologies: {', '.join(exp['technologies'][:5])}")
        print()


def demonstrate_project_recommendations():
    """Demonstrate project recommendation analysis."""
    print("=== PROJECT RECOMMENDATIONS ===\n")

    matcher = ProfileMatcher()
    job_data = create_sample_job_data()
    user_profile = create_sample_user_profile()

    project_recommendations = matcher.recommend_projects(user_profile.projects, job_data)

    print(f"🚀 Recommended Projects (Top {len(project_recommendations)}):")
    for i, proj in enumerate(project_recommendations, 1):
        print(f"   {i}. {proj.title}")
        print(f"      Relevance Score: {proj.relevance_score:.2f}")
        print(f"      Matching Technologies: {', '.join(proj.matching_technologies)}")
        print(f"      Why Relevant: {proj.relevance_reason}")
        if proj.url:
            print(f"      URL: {proj.url}")
        print()


def demonstrate_full_matching_analysis():
    """Demonstrate complete profile-job matching analysis."""
    print("=== COMPREHENSIVE PROFILE-JOB MATCHING ===\n")

    try:
        # Initialize matcher
        matcher = ProfileMatcher()

        # Create sample data
        job_data = create_sample_job_data()
        user_profile = create_sample_user_profile()

        print(f"🎯 Analyzing compatibility for:")
        print(f"   Position: {job_data.position} at {job_data.company_name}")
        print(f"   Candidate: {user_profile.personal_info.name}")
        print(f"   Experience: {user_profile.get_total_experience_years()} years")
        print()

        # Perform comprehensive analysis
        print("🔄 Running AI-powered analysis...")
        match_result = matcher.match_profile(job_data, user_profile)

        # Display results
        print(f"📈 MATCH ANALYSIS RESULTS")
        print(f"   Overall Score: {match_result.score}/10 ({match_result.get_match_category()})")
        print(f"   Confidence Level: {match_result.confidence_level:.1%}")
        print()

        print(f"✅ Matched Skills ({len(match_result.matched_skills)}):")
        print(f"   {', '.join(match_result.matched_skills[:10])}")
        print()

        details = match_result.analysis_details
        print(f"📊 Detailed Analysis:")
        print(f"   Skills Match: {details.skills_match_percentage:.1f}%")
        print(f"   Experience Match: {details.experience_match_percentage:.1f}%")
        print(f"   Education Match: {'✓' if details.education_match else '✗'}")
        print()

        if details.strength_areas:
            print(f"💪 Strengths:")
            for strength in details.strength_areas[:5]:
                print(f"   • {strength}")
            print()

        if details.weakness_areas:
            print(f"⚠️  Areas for Improvement:")
            for weakness in details.weakness_areas[:5]:
                print(f"   • {weakness}")
            print()

        if match_result.relevant_experiences:
            print(f"🏢 Most Relevant Experiences:")
            for exp in match_result.relevant_experiences[:3]:
                print(f"   • {exp.role} at {exp.company} (Score: {exp.relevance_score:.2f})")
            print()

        if match_result.recommended_projects:
            print(f"🚀 Recommended Projects to Highlight:")
            for proj in match_result.recommended_projects[:3]:
                print(f"   • {proj.title} (Relevance: {proj.relevance_score:.2f})")
            print()

        # Generate improvement suggestions
        suggestions = matcher.generate_improvement_suggestions(match_result)
        if suggestions:
            print(f"💡 Improvement Suggestions:")
            for suggestion in suggestions[:4]:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(suggestion.priority, "⚪")
                print(f"   {priority_emoji} {suggestion.suggestion[:100]}...")
                if suggestion.impact:
                    print(f"      Impact: {suggestion.impact}")
                print()

        # Summary recommendation
        print(f"🎯 RECOMMENDATION:")
        if match_result.score >= 8:
            print("   STRONG MATCH - Apply with confidence! Focus on highlighting your most relevant experiences.")
        elif match_result.score >= 6:
            print("   GOOD MATCH - Strong candidate with some areas for improvement. Consider addressing key gaps.")
        elif match_result.score >= 4:
            print("   MODERATE MATCH - Consider developing missing skills or targeting more suitable roles.")
        else:
            print("   WEAK MATCH - Significant skill and experience gaps. Focus on professional development first.")

    except ProfileMatchingError as e:
        print(f"❌ Error during analysis: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Main example execution."""
    print("🤖 ProfileMatcher Example - AI-Powered Job-Profile Compatibility Analysis\n")
    print("=" * 80)

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   Set your OpenAI API key to run the full analysis:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()

        # Run non-AI dependent examples
        demonstrate_skill_matching()
        demonstrate_experience_selection()
        demonstrate_project_recommendations()

        print("🔒 Full AI analysis requires OpenAI API key. Set OPENAI_API_KEY to continue.")
        return

    # Run all demonstrations
    try:
        demonstrate_skill_matching()
        demonstrate_experience_selection()
        demonstrate_project_recommendations()
        demonstrate_full_matching_analysis()

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("   Make sure your OpenAI API key is valid and you have sufficient credits.")


if __name__ == "__main__":
    main()