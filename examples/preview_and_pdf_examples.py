"""
Comprehensive examples demonstrating the PreviewGenerator and PDFGenerator functionality.

This module provides complete usage examples showing how to use the preview and PDF
generation system with various scenarios and error handling patterns.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Import the modules (adjust imports based on your project structure)
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.preview_generator import PreviewGenerator, PreviewOptions
from src.agents.pdf_generator import PDFGenerator, PDFOptions, generate_application_pdfs_sync
from src.models.job_data import JobData, ExperienceLevel
from src.models.match_result import (
    MatchResult, MatchScore, AnalysisDetails, RelevantExperience,
    RecommendedProject, ImprovementSuggestion, SuggestionCategory
)

# Set up logging for examples
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_job_data() -> JobData:
    """Create sample job data for examples."""
    return JobData(
        company_name="TechCorp Solutions",
        position="Senior Python Developer",
        requirements=[
            "5+ years of Python development experience",
            "Experience with web frameworks (Django, Flask)",
            "Knowledge of cloud platforms (AWS, Azure)",
            "Strong problem-solving skills",
            "Experience with agile development"
        ],
        skills_required=[
            "Python", "Django", "Flask", "PostgreSQL", "AWS",
            "Docker", "Git", "REST APIs", "Microservices"
        ],
        experience_level=ExperienceLevel.SENIOR,
        description="""
        We are seeking a Senior Python Developer to join our growing engineering team.
        The ideal candidate will have extensive experience in Python web development,
        cloud technologies, and building scalable applications. You will work on
        challenging projects that impact millions of users worldwide.

        Responsibilities include designing and implementing new features, optimizing
        existing systems, mentoring junior developers, and collaborating with
        cross-functional teams to deliver high-quality software solutions.
        """,
        url="https://techcorp.com/careers/senior-python-developer"
    )


def create_sample_match_result() -> MatchResult:
    """Create sample match result for examples."""
    analysis_details = AnalysisDetails(
        skills_match_percentage=85.5,
        experience_match_percentage=78.0,
        education_match=True,
        missing_skills=["Kubernetes", "GraphQL"],
        additional_skills=["Machine Learning", "Data Science", "React"],
        experience_gap_months=0,
        strength_areas=["Python Development", "Web Frameworks", "Cloud Platforms"],
        weakness_areas=["Container Orchestration", "API Design"],
        match_explanation="Strong match with excellent Python skills and relevant experience"
    )

    relevant_experiences = [
        RelevantExperience(
            company="Previous Tech Co",
            role="Python Developer",
            relevance_score=0.92,
            matching_skills=["Python", "Django", "PostgreSQL", "AWS"],
            key_achievements=[
                "Built scalable microservices handling 1M+ requests daily",
                "Reduced API response time by 40% through optimization",
                "Led migration to cloud infrastructure"
            ],
            duration_months=36,
            relevance_reason="Direct Python development experience with matching technologies"
        ),
        RelevantExperience(
            company="StartupXYZ",
            role="Full Stack Developer",
            relevance_score=0.78,
            matching_skills=["Python", "Flask", "Docker"],
            key_achievements=[
                "Developed MVP that secured $2M funding",
                "Implemented CI/CD pipeline",
                "Built real-time analytics dashboard"
            ],
            duration_months=24,
            relevance_reason="Full-stack experience with Python backend development"
        )
    ]

    recommended_projects = [
        RecommendedProject(
            title="E-commerce Platform",
            relevance_score=0.89,
            matching_technologies=["Python", "Django", "PostgreSQL", "Redis"],
            description="Built scalable e-commerce platform serving 100K+ users",
            relevance_reason="Demonstrates large-scale Python web development skills",
            url="https://github.com/user/ecommerce-platform"
        ),
        RecommendedProject(
            title="API Gateway Service",
            relevance_score=0.83,
            matching_technologies=["Python", "Flask", "Docker", "AWS"],
            description="Microservices API gateway with rate limiting and monitoring",
            relevance_reason="Shows microservices architecture and cloud deployment experience"
        )
    ]

    suggestions = [
        ImprovementSuggestion(
            category=SuggestionCategory.SKILLS,
            priority="high",
            suggestion="Learn Kubernetes for container orchestration to strengthen DevOps skills",
            impact="Would make you competitive for senior roles requiring cloud-native expertise",
            resources=["Kubernetes Documentation", "Cloud Native Computing Foundation courses"]
        ),
        ImprovementSuggestion(
            category=SuggestionCategory.EXPERIENCE,
            priority="medium",
            suggestion="Gain experience with GraphQL APIs to expand backend capabilities",
            impact="Modern API development skills highly valued in current market",
            resources=["GraphQL official tutorial", "Apollo GraphQL courses"]
        )
    ]

    return MatchResult(
        score=MatchScore.VERY_GOOD,
        matched_skills=["Python", "Django", "Flask", "PostgreSQL", "AWS", "Docker", "Git"],
        relevant_experiences=relevant_experiences,
        recommended_projects=recommended_projects,
        suggestions=suggestions,
        analysis_details=analysis_details,
        job_title="Senior Python Developer",
        company_name="TechCorp Solutions",
        analyzed_at=datetime.now(),
        confidence_level=0.87
    )


def create_sample_html_content() -> Dict[str, str]:
    """Create sample HTML content for CV and cover letter."""
    cv_html = """
    <div class="cv-container">
        <header class="header">
            <h1>John Smith</h1>
            <div class="contact-info">
                <p>Email: john.smith@email.com | Phone: (555) 123-4567</p>
                <p>LinkedIn: linkedin.com/in/johnsmith | GitHub: github.com/johnsmith</p>
            </div>
        </header>

        <section class="section">
            <h2>Professional Summary</h2>
            <p>Experienced Python Developer with 6+ years of expertise in web development,
            cloud technologies, and scalable system design. Proven track record of delivering
            high-quality software solutions in fast-paced environments.</p>
        </section>

        <section class="section">
            <h2>Technical Skills</h2>
            <ul class="skills-list">
                <li><strong>Programming Languages:</strong> Python, JavaScript, SQL</li>
                <li><strong>Web Frameworks:</strong> Django, Flask, FastAPI</li>
                <li><strong>Databases:</strong> PostgreSQL, Redis, MongoDB</li>
                <li><strong>Cloud Platforms:</strong> AWS (EC2, S3, RDS, Lambda)</li>
                <li><strong>Tools & Technologies:</strong> Docker, Git, Jenkins, Kubernetes</li>
            </ul>
        </section>

        <section class="section">
            <h2>Professional Experience</h2>
            <div class="experience-item">
                <div class="job-title">Senior Python Developer</div>
                <div class="company-name">Previous Tech Co</div>
                <div class="date-range">January 2021 - Present</div>
                <ul class="achievements">
                    <li>Built scalable microservices handling 1M+ requests daily</li>
                    <li>Reduced API response time by 40% through optimization</li>
                    <li>Led migration to cloud infrastructure, reducing costs by 30%</li>
                    <li>Mentored 3 junior developers and established code review processes</li>
                </ul>
            </div>

            <div class="experience-item">
                <div class="job-title">Full Stack Developer</div>
                <div class="company-name">StartupXYZ</div>
                <div class="date-range">March 2019 - December 2020</div>
                <ul class="achievements">
                    <li>Developed MVP that secured $2M funding round</li>
                    <li>Implemented CI/CD pipeline reducing deployment time by 75%</li>
                    <li>Built real-time analytics dashboard using Django and React</li>
                </ul>
            </div>
        </section>

        <section class="section">
            <h2>Education</h2>
            <div class="education-item">
                <div class="degree">Bachelor of Science in Computer Science</div>
                <div class="institution">University of Technology</div>
                <div class="date-range">2015 - 2019</div>
            </div>
        </section>

        <section class="section">
            <h2>Key Projects</h2>
            <div class="project-item">
                <div class="project-title">E-commerce Platform</div>
                <p>Built scalable e-commerce platform serving 100K+ users using Django, PostgreSQL, and Redis.</p>
            </div>
            <div class="project-item">
                <div class="project-title">API Gateway Service</div>
                <p>Microservices API gateway with rate limiting and monitoring, deployed on AWS using Docker.</p>
            </div>
        </section>
    </div>
    """

    cover_letter_html = """
    <div class="letter-container">
        <div class="letter-header">
            <div class="sender-info">
                <h3>John Smith</h3>
                <p>Email: john.smith@email.com</p>
                <p>Phone: (555) 123-4567</p>
            </div>
        </div>

        <div class="letter-date">
            <p>December 15, 2024</p>
        </div>

        <div class="letter-address">
            <p>Hiring Manager<br>
            TechCorp Solutions<br>
            123 Technology Drive<br>
            San Francisco, CA 94105</p>
        </div>

        <div class="letter-greeting">
            <p>Dear Hiring Manager,</p>
        </div>

        <div class="letter-body">
            <p>I am writing to express my strong interest in the Senior Python Developer position
            at TechCorp Solutions. With over 6 years of experience in Python development and a
            proven track record of building scalable web applications, I am excited about the
            opportunity to contribute to your innovative engineering team.</p>

            <p>In my current role at Previous Tech Co, I have successfully built microservices
            handling over 1 million requests daily and reduced API response times by 40% through
            strategic optimization. My experience with Django, Flask, and cloud platforms like
            AWS directly aligns with your requirements for this position.</p>

            <p>What particularly excites me about TechCorp Solutions is your commitment to
            building products that impact millions of users. I thrive in challenging environments
            where I can apply my technical expertise to solve complex problems while mentoring
            junior developers and fostering a collaborative team culture.</p>

            <p>I would welcome the opportunity to discuss how my experience and passion for
            Python development can contribute to TechCorp's continued success. Thank you for
            considering my application.</p>
        </div>

        <div class="letter-closing">
            <p>Sincerely,</p>
            <div class="signature-space"></div>
            <p>John Smith</p>
        </div>
    </div>
    """

    return {
        'cv': cv_html,
        'cover_letter': cover_letter_html
    }


def create_sample_original_content() -> Dict[str, str]:
    """Create sample original content for comparison."""
    return {
        'cv': """
        <div class="cv-container">
            <header>
                <h1>John Smith</h1>
                <p>Email: john.smith@email.com | Phone: (555) 123-4567</p>
            </header>

            <section>
                <h2>Professional Summary</h2>
                <p>Python Developer with experience in web development.</p>
            </section>

            <section>
                <h2>Experience</h2>
                <div>
                    <h3>Python Developer - Previous Tech Co</h3>
                    <p>Worked on various Python projects.</p>
                </div>
            </section>
        </div>
        """,
        'cover_letter': """
        <div class="letter-container">
            <p>Dear Hiring Manager,</p>
            <p>I am interested in the Python Developer position.</p>
            <p>Thank you for your consideration.</p>
            <p>Sincerely, John Smith</p>
        </div>
        """
    }


def example_1_basic_preview_generation():
    """Example 1: Basic preview generation with all features."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Preview Generation")
    print("="*80)

    try:
        # Create sample data
        job_data = create_sample_job_data()
        match_result = create_sample_match_result()
        customized_html = create_sample_html_content()
        original_html = create_sample_original_content()

        changes = [
            "Enhanced professional summary to highlight 6+ years of experience",
            "Added specific technical achievements with quantifiable metrics",
            "Emphasized cloud platform expertise to match job requirements",
            "Included mentoring experience to demonstrate leadership skills",
            "Customized cover letter to mention company's mission and values"
        ]

        # Initialize preview generator with custom options
        options = PreviewOptions(
            show_side_by_side=True,
            include_match_analysis=True,
            enable_interactive_approval=True,
            highlight_changes=True,
            theme="professional"
        )

        generator = PreviewGenerator(options)

        # Generate preview
        preview_html = generator.generate_preview(
            customized_html=customized_html,
            match_result=match_result,
            changes=changes,
            original_html=original_html
        )

        # Save preview to file for viewing
        output_path = Path(__file__).parent / "output" / "example_1_preview.html"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(preview_html)

        print(f"✅ Preview generated successfully!")
        print(f"📁 Saved to: {output_path}")
        print(f"🌐 Open the file in a browser to view the interactive preview")
        print(f"📊 Match Score: {match_result.score}/10 ({match_result.get_match_category()})")
        print(f"📝 Number of customizations: {len(changes)}")

    except Exception as e:
        print(f"❌ Error in basic preview generation: {e}")
        logger.exception("Example 1 failed")


def example_2_side_by_side_comparison():
    """Example 2: Focus on side-by-side comparison functionality."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Side-by-Side Comparison")
    print("="*80)

    try:
        generator = PreviewGenerator()
        original_html = create_sample_original_content()
        customized_html = create_sample_html_content()

        # Generate comparison HTML
        comparison_html = generator.create_side_by_side_comparison(
            original=original_html,
            customized=customized_html
        )

        # Create a minimal HTML document with just the comparison
        full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Side-by-Side Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .comparison-section {{ margin-bottom: 30px; background: white; border-radius: 8px; overflow: hidden; }}
        .comparison-section h3 {{ background: #007bff; color: white; padding: 15px; margin: 0; }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
        .original-content {{ padding: 20px; background: #f8f9fa; border-right: 1px solid #dee2e6; }}
        .customized-content {{ padding: 20px; background: #e8f5e8; }}
        .original-content h4, .customized-content h4 {{ margin-top: 0; color: #495057; }}
        .content-box {{ background: white; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; }}
        .highlighted {{ border-color: #28a745; box-shadow: 0 0 5px rgba(40, 167, 69, 0.3); }}
    </style>
</head>
<body>
    <h1>Document Comparison Example</h1>
    {comparison_html}
</body>
</html>
        """

        # Save comparison to file
        output_path = Path(__file__).parent / "output" / "example_2_comparison.html"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)

        print(f"✅ Side-by-side comparison generated successfully!")
        print(f"📁 Saved to: {output_path}")
        print(f"🔍 View the file to see original vs customized content comparison")

    except Exception as e:
        print(f"❌ Error in comparison generation: {e}")
        logger.exception("Example 2 failed")


def example_3_pdf_generation():
    """Example 3: PDF generation with various options."""
    print("\n" + "="*80)
    print("EXAMPLE 3: PDF Generation")
    print("="*80)

    try:
        # Create sample data
        job_data = create_sample_job_data()
        customized_html = create_sample_html_content()

        # Test different PDF options
        pdf_options = PDFOptions(
            format="A4",
            margin_top="1.5cm",
            margin_bottom="1.5cm",
            margin_left="2cm",
            margin_right="2cm",
            print_background=True,
            scale=0.9,
            timeout=45000  # 45 seconds
        )

        # Generate PDFs using synchronous wrapper
        print("🔄 Generating PDFs...")
        pdf_paths = generate_application_pdfs_sync(
            cv_html=customized_html['cv'],
            cover_letter_html=customized_html['cover_letter'],
            job_data=job_data,
            output_folder=None,  # Will auto-generate folder
            options=pdf_options
        )

        print(f"✅ PDFs generated successfully!")
        for doc_type, path in pdf_paths.items():
            print(f"📄 {doc_type.upper()}: {path}")

            # Validate the PDF
            generator = PDFGenerator()
            is_valid = generator.validate_pdf_output(path)
            print(f"   ✓ Validation: {'PASSED' if is_valid else 'FAILED'}")

            # Check file size
            file_size = Path(path).stat().st_size
            print(f"   📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    except Exception as e:
        print(f"❌ Error in PDF generation: {e}")
        logger.exception("Example 3 failed")


async def example_4_async_pdf_generation():
    """Example 4: Asynchronous PDF generation with context management."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Async PDF Generation")
    print("="*80)

    try:
        # Create sample data
        job_data = create_sample_job_data()
        customized_html = create_sample_html_content()

        # Use async context manager for better resource management
        pdf_options = PDFOptions(
            format="A4",
            margin_top="1cm",
            margin_bottom="1cm",
            generate_tagged_pdf=True,  # For accessibility
            wait_for_timeout=3000
        )

        async with PDFGenerator(pdf_options) as generator:
            print("🔄 Generating PDFs asynchronously...")

            # Generate both PDFs
            pdf_paths = await generator.generate_pdfs(
                approved_html=customized_html,
                job_data=job_data
            )

            print(f"✅ Async PDF generation completed!")
            for doc_type, path in pdf_paths.items():
                print(f"📄 {doc_type.upper()}: {path}")

                # Validate each PDF
                is_valid = generator.validate_pdf_output(path)
                print(f"   ✓ Validation: {'PASSED' if is_valid else 'FAILED'}")

        print("🧹 Browser resources cleaned up automatically")

    except Exception as e:
        print(f"❌ Error in async PDF generation: {e}")
        logger.exception("Example 4 failed")


def example_5_error_handling():
    """Example 5: Comprehensive error handling scenarios."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Error Handling Scenarios")
    print("="*80)

    # Test 1: Invalid input validation
    print("\n🧪 Test 1: Input validation")
    try:
        generator = PreviewGenerator()
        # Invalid input - missing required keys
        invalid_html = {'invalid_key': 'content'}
        match_result = create_sample_match_result()

        generator.generate_preview(invalid_html, match_result, [])
    except Exception as e:
        print(f"✅ Caught expected validation error: {type(e).__name__}: {e}")

    # Test 2: Empty content handling
    print("\n🧪 Test 2: Empty content handling")
    try:
        generator = PreviewGenerator()
        empty_html = {'cv': '', 'cover_letter': ''}
        match_result = create_sample_match_result()

        preview = generator.generate_preview(empty_html, match_result, [])
        print("✅ Successfully handled empty content")
    except Exception as e:
        print(f"⚠️  Error with empty content: {e}")

    # Test 3: PDF generation with invalid HTML
    print("\n🧪 Test 3: PDF generation error handling")
    try:
        job_data = create_sample_job_data()
        invalid_html = {'cv': '<invalid><html', 'cover_letter': 'test'}

        # This should handle the invalid HTML gracefully
        generator = PDFGenerator()
        paths = generator.generate_pdfs_sync(invalid_html, job_data)
        print("✅ PDF generation handled invalid HTML")
    except Exception as e:
        print(f"✅ Caught expected PDF error: {type(e).__name__}: {e}")

    # Test 4: File permission errors
    print("\n🧪 Test 4: File permission simulation")
    try:
        # Try to write to a non-existent directory path
        generator = PDFGenerator()
        result = generator.validate_pdf_output("/invalid/path/test.pdf")
        print(f"✅ Validation correctly failed: {result}")
    except Exception as e:
        print(f"✅ Handled file access error: {e}")


def example_6_complete_workflow():
    """Example 6: Complete workflow from preview to PDF generation."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Complete Workflow")
    print("="*80)

    try:
        # Step 1: Create all necessary data
        print("📋 Step 1: Preparing application data...")
        job_data = create_sample_job_data()
        match_result = create_sample_match_result()
        customized_html = create_sample_html_content()
        original_html = create_sample_original_content()

        changes = [
            "Tailored professional summary for the specific role",
            "Highlighted relevant Python and cloud experience",
            "Added quantifiable achievements and metrics",
            "Emphasized leadership and mentoring capabilities",
            "Customized cover letter tone and content for company culture"
        ]

        # Step 2: Generate preview
        print("👀 Step 2: Generating interactive preview...")
        preview_generator = PreviewGenerator(PreviewOptions(
            show_side_by_side=True,
            include_match_analysis=True,
            enable_interactive_approval=True
        ))

        preview_html = preview_generator.generate_preview(
            customized_html=customized_html,
            match_result=match_result,
            changes=changes,
            original_html=original_html
        )

        # Save preview
        preview_path = Path(__file__).parent / "output" / "complete_workflow_preview.html"
        preview_path.parent.mkdir(exist_ok=True)

        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(preview_html)

        print(f"✅ Preview saved: {preview_path}")

        # Step 3: Simulate user approval (in real scenario, this would be interactive)
        print("👤 Step 3: Simulating user approval...")
        user_approved = preview_generator.get_user_approval()  # Returns True (placeholder)

        if user_approved:
            print("✅ User approved the application materials")

            # Step 4: Generate final PDFs
            print("📄 Step 4: Generating final PDFs...")
            pdf_options = PDFOptions(
                format="A4",
                margin_top="1.5cm",
                margin_bottom="1.5cm",
                print_background=True,
                scale=0.95
            )

            pdf_paths = generate_application_pdfs_sync(
                cv_html=customized_html['cv'],
                cover_letter_html=customized_html['cover_letter'],
                job_data=job_data,
                options=pdf_options
            )

            # Step 5: Final validation and summary
            print("✅ Step 5: Final validation and summary...")
            for doc_type, path in pdf_paths.items():
                generator = PDFGenerator()
                is_valid = generator.validate_pdf_output(path)
                file_size = Path(path).stat().st_size

                print(f"📄 {doc_type.upper()}: {Path(path).name}")
                print(f"   📁 Location: {path}")
                print(f"   ✓ Valid: {'YES' if is_valid else 'NO'}")
                print(f"   📊 Size: {file_size:,} bytes")

            # Summary
            print(f"\n📊 WORKFLOW SUMMARY:")
            print(f"   🏢 Company: {job_data.company_name}")
            print(f"   💼 Position: {job_data.position}")
            print(f"   📈 Match Score: {match_result.score}/10")
            print(f"   🔧 Customizations: {len(changes)}")
            print(f"   📄 PDFs Generated: {len(pdf_paths)}")
            print(f"   👀 Preview Available: Yes")

        else:
            print("❌ User rejected the application materials")

    except Exception as e:
        print(f"❌ Error in complete workflow: {e}")
        logger.exception("Example 6 failed")


def run_all_examples():
    """Run all examples in sequence."""
    print("🚀 Starting Preview and PDF Generation Examples")
    print("=" * 80)

    examples = [
        example_1_basic_preview_generation,
        example_2_side_by_side_comparison,
        example_3_pdf_generation,
        example_5_error_handling,  # Skip async example in sync context
        example_6_complete_workflow
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            logger.exception(f"Example {i} failed")

        # Small delay between examples
        import time
        time.sleep(1)

    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("📁 Check the 'output' folder for generated files")
    print("🌐 Open the HTML files in a browser to see the previews")
    print("📄 PDF files can be opened with any PDF viewer")


async def run_async_examples():
    """Run async examples."""
    print("🚀 Running Async Examples")
    await example_4_async_pdf_generation()


if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Run synchronous examples
    run_all_examples()

    # Run async examples
    print("\n" + "="*80)
    print("🔄 Running Async Examples...")
    try:
        asyncio.run(run_async_examples())
    except Exception as e:
        print(f"❌ Async examples failed: {e}")

    print("\n🎉 All examples completed successfully!")
    print("📖 Check the generated files in the 'output' directory")