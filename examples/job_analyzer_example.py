#!/usr/bin/env python3
"""
Example usage of the JobAnalyzer class.

This script demonstrates how to use the JobAnalyzer for analyzing job postings
from both URLs and raw text input. It shows typical usage patterns and error
handling approaches.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.job_analyzer import JobAnalyzer, JobAnalysisError, OpenAIConfigurationError
from src.models.job_data import JobData


def main():
    """Main function demonstrating JobAnalyzer usage."""
    print("=== JobAnalyzer Example Usage ===\n")

    # Initialize the analyzer
    try:
        print("1. Initializing JobAnalyzer...")
        analyzer = JobAnalyzer()
        print("✓ JobAnalyzer initialized successfully\n")

    except OpenAIConfigurationError as e:
        print(f"✗ Configuration error: {e}")
        print("Please set the OPENAI_API_KEY environment variable.")
        return 1

    # Example 1: Analyze job posting from text
    print("2. Analyzing job posting from text...")
    sample_job_text = """
    Senior Python Developer
    TechCorp Solutions

    We are seeking a Senior Python Developer to join our dynamic team.

    Requirements:
    • 5+ years of Python development experience
    • Strong knowledge of Django and Flask frameworks
    • Experience with PostgreSQL and MongoDB
    • Familiarity with AWS cloud services
    • Bachelor's degree in Computer Science or equivalent

    Responsibilities:
    • Develop and maintain web applications
    • Lead technical discussions and code reviews
    • Mentor junior developers
    • Collaborate with product and design teams

    Skills Required:
    Python, Django, Flask, PostgreSQL, MongoDB, AWS, Git, REST APIs

    We offer competitive salary, remote work options, and comprehensive benefits.
    """

    try:
        job_data = analyzer.analyze_job(sample_job_text)
        print("✓ Job analysis completed successfully!")
        print(f"  Company: {job_data.company_name}")
        print(f"  Position: {job_data.position}")
        print(f"  Experience Level: {job_data.experience_level.value}")
        print(f"  Skills: {', '.join(job_data.skills_required[:5])}...")
        print(f"  Requirements Count: {len(job_data.requirements)}")
        print()

        # Generate analysis summary
        summary = analyzer.get_analysis_summary(job_data)
        print("📊 Analysis Summary:")
        print(f"  Validation Passed: {summary['validation_passed']}")
        print(f"  Skills Count: {summary['skills_count']}")
        print(f"  Description Length: {summary['description_length']} characters")
        print()

    except JobAnalysisError as e:
        print(f"✗ Job analysis failed: {e}\n")

    # Example 2: Analyze job posting from URL (commented out for safety)
    print("3. Analyzing job posting from URL...")
    print("(Skipped in example - requires valid job posting URL)")
    """
    job_url = "https://example.com/job-posting"
    try:
        job_data = analyzer.analyze_job(job_url)
        print(f"✓ Successfully analyzed job from URL: {job_url}")
        print(f"  Company: {job_data.company_name}")
        print(f"  Position: {job_data.position}")
        print()

    except JobAnalysisError as e:
        print(f"✗ Failed to analyze job from URL: {e}")
        print("  This could be due to scraping restrictions or invalid URL")
        print()
    """

    # Example 3: Data validation
    print("4. Demonstrating data validation...")
    try:
        # Create job data with potential quality issues
        test_job_text = "Short job description."
        job_data = analyzer.extract_job_details(test_job_text)

        # Validate the extraction
        is_valid = analyzer.validate_extraction(job_data)
        print(f"  Validation result: {'✓ Passed' if is_valid else '⚠ Failed'}")

        if not is_valid:
            print("  This indicates potential quality issues with the extracted data")

    except JobAnalysisError as e:
        print(f"✗ Validation example failed: {e}")

    print()

    # Example 4: Error handling
    print("5. Demonstrating error handling...")

    # Test with empty input
    try:
        analyzer.analyze_job("")
    except JobAnalysisError as e:
        print(f"✓ Correctly handled empty input: {type(e).__name__}")

    # Test with very short input
    try:
        analyzer.analyze_job("Hi")
    except JobAnalysisError as e:
        print(f"✓ Correctly handled short input: {type(e).__name__}")

    print()

    # Example 5: Skill matching
    print("6. Skill matching example...")
    if 'job_data' in locals():
        user_skills = ["Python", "Django", "JavaScript", "React", "PostgreSQL"]
        matches = job_data.get_skill_matches(user_skills)
        print(f"  User skills: {user_skills}")
        print(f"  Matching skills: {matches}")
        print(f"  Has skill match: {job_data.is_skill_match(user_skills)}")

    print("\n=== Example completed ===")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)