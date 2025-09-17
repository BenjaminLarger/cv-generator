#!/usr/bin/env python3
"""
Integration Demo for Preview and PDF Generation System

This script demonstrates the complete workflow of the preview and PDF generation
system, including error handling, file organization, and user interaction simulation.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.preview_generator import PreviewGenerator, PreviewOptions
from src.agents.pdf_generator import PDFGenerator, PDFOptions, generate_application_pdfs_sync
from src.models.job_data import JobData, ExperienceLevel
from src.models.match_result import (
    MatchResult, MatchScore, AnalysisDetails, RelevantExperience,
    RecommendedProject, ImprovementSuggestion, SuggestionCategory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_data():
    """Create comprehensive demo data for the integration test."""

    # Create job data
    job_data = JobData(
        company_name="InnovateTech Solutions",
        position="Senior Full Stack Developer",
        requirements=[
            "5+ years of full-stack development experience",
            "Expert knowledge of Python and Django",
            "Experience with React and modern JavaScript",
            "Strong database design skills (PostgreSQL, Redis)",
            "Cloud platform experience (AWS preferred)",
            "Experience with Docker and Kubernetes",
            "Understanding of microservices architecture",
            "Strong communication and leadership skills"
        ],
        skills_required=[
            "Python", "Django", "React", "JavaScript", "PostgreSQL",
            "Redis", "AWS", "Docker", "Kubernetes", "REST APIs",
            "GraphQL", "Git", "CI/CD", "Microservices"
        ],
        experience_level=ExperienceLevel.SENIOR,
        description="""
        We are seeking a Senior Full Stack Developer to join our innovative engineering team
        at InnovateTech Solutions. You will be responsible for designing and implementing
        scalable web applications that serve millions of users worldwide.

        As a senior team member, you will lead technical initiatives, mentor junior developers,
        and work closely with product managers to deliver high-quality software solutions.
        Our tech stack includes Python/Django for backend services, React for frontend
        applications, and AWS for cloud infrastructure.

        This is an excellent opportunity to work on cutting-edge projects in a fast-paced,
        collaborative environment where your technical expertise will directly impact
        business growth and user experience.
        """,
        url="https://innovatetech.com/careers/senior-fullstack-developer"
    )

    # Create analysis details
    analysis_details = AnalysisDetails(
        skills_match_percentage=88.5,
        experience_match_percentage=92.0,
        education_match=True,
        missing_skills=["Kubernetes", "GraphQL"],
        additional_skills=["Machine Learning", "Data Science", "Vue.js"],
        experience_gap_months=0,
        strength_areas=[
            "Python/Django Development",
            "Full-Stack Architecture",
            "Cloud Infrastructure",
            "Team Leadership"
        ],
        weakness_areas=[
            "Container Orchestration",
            "Modern API Design",
            "DevOps Practices"
        ],
        match_explanation="""
        Excellent match with strong alignment in core technologies and experience level.
        The candidate demonstrates comprehensive full-stack capabilities with Python/Django
        backend expertise and modern frontend skills. Leadership experience and cloud
        platform knowledge make this a highly competitive profile for the role.
        """
    )

    # Create relevant experiences
    relevant_experiences = [
        RelevantExperience(
            company="TechCorp Industries",
            role="Senior Python Developer",
            relevance_score=0.94,
            matching_skills=["Python", "Django", "PostgreSQL", "AWS", "React"],
            key_achievements=[
                "Led development of microservices architecture serving 2M+ daily users",
                "Reduced API response times by 65% through database optimization",
                "Mentored team of 6 developers and established code review processes",
                "Implemented CI/CD pipeline reducing deployment time from hours to minutes"
            ],
            duration_months=42,
            relevance_reason="""
            Direct experience in senior role with matching technology stack.
            Demonstrates leadership, scalability expertise, and performance optimization skills.
            """
        ),
        RelevantExperience(
            company="StartupXYZ",
            role="Full Stack Engineer",
            relevance_score=0.87,
            matching_skills=["Python", "Django", "React", "JavaScript", "PostgreSQL"],
            key_achievements=[
                "Built MVP from scratch that secured $5M Series A funding",
                "Developed real-time collaborative features using WebSockets",
                "Implemented comprehensive testing suite with 95% coverage",
                "Designed and deployed scalable database architecture"
            ],
            duration_months=30,
            relevance_reason="""
            Comprehensive full-stack experience building scalable applications from the ground up.
            Strong foundation in both backend and frontend technologies.
            """
        ),
        RelevantExperience(
            company="CloudSys Solutions",
            role="Backend Developer",
            relevance_score=0.79,
            matching_skills=["Python", "Django", "AWS", "Docker", "REST APIs"],
            key_achievements=[
                "Migrated legacy monolith to cloud-native microservices",
                "Implemented robust API gateway with rate limiting and monitoring",
                "Reduced infrastructure costs by 40% through AWS optimization",
                "Built automated deployment pipeline using Docker and AWS ECS"
            ],
            duration_months=24,
            relevance_reason="""
            Strong backend and cloud experience directly applicable to the role.
            Demonstrates expertise in cloud migration and microservices architecture.
            """
        )
    ]

    # Create recommended projects
    recommended_projects = [
        RecommendedProject(
            title="E-Commerce Platform with Microservices",
            relevance_score=0.92,
            matching_technologies=["Python", "Django", "React", "PostgreSQL", "AWS", "Docker"],
            description="""
            Built comprehensive e-commerce platform serving 500K+ active users with
            microservices architecture. Features include real-time inventory management,
            payment processing, order tracking, and analytics dashboard.
            """,
            relevance_reason="""
            Demonstrates full-stack capabilities with microservices architecture
            and high-scale user management - directly relevant to the position.
            """,
            url="https://github.com/johndoe/ecommerce-microservices"
        ),
        RecommendedProject(
            title="Real-Time Collaboration Tool",
            relevance_score=0.85,
            matching_technologies=["Python", "Django", "React", "WebSockets", "Redis"],
            description="""
            Developed real-time collaborative workspace with document editing,
            video conferencing, and team chat features. Supports concurrent
            users with conflict resolution and synchronization.
            """,
            relevance_reason="""
            Shows expertise in real-time systems and complex frontend interactions.
            Relevant for building interactive user experiences.
            """,
            url="https://github.com/johndoe/collab-workspace"
        ),
        RecommendedProject(
            title="API Gateway and Service Mesh",
            relevance_score=0.88,
            matching_technologies=["Python", "Docker", "Kubernetes", "AWS", "Microservices"],
            description="""
            Implemented enterprise-grade API gateway with service discovery,
            load balancing, authentication, and monitoring. Deployed using
            Kubernetes with auto-scaling capabilities.
            """,
            relevance_reason="""
            Demonstrates advanced microservices architecture and cloud-native
            development skills essential for senior-level positions.
            """,
            url="https://github.com/johndoe/api-gateway-mesh"
        )
    ]

    # Create improvement suggestions
    suggestions = [
        ImprovementSuggestion(
            category=SuggestionCategory.SKILLS,
            priority="high",
            suggestion="""
            Gain hands-on experience with Kubernetes for container orchestration.
            Focus on deployment strategies, service mesh concepts, and cluster management.
            """,
            impact="""
            Kubernetes expertise is increasingly important for senior roles in cloud-native
            environments. This skill would make you competitive for technical leadership positions.
            """,
            resources=[
                "Kubernetes Official Documentation",
                "Cloud Native Computing Foundation (CNCF) Training",
                "Kubernetes the Hard Way tutorial",
                "AWS EKS or Google GKE hands-on labs"
            ]
        ),
        ImprovementSuggestion(
            category=SuggestionCategory.SKILLS,
            priority="medium",
            suggestion="""
            Learn GraphQL for modern API development and client-server communication.
            Understand schema design, resolvers, and performance optimization.
            """,
            impact="""
            GraphQL is becoming the standard for flexible API development.
            This knowledge would enhance your full-stack capabilities significantly.
            """,
            resources=[
                "GraphQL official tutorial",
                "Apollo GraphQL documentation",
                "GraphQL with Python and Django integration guides",
                "Building efficient GraphQL schemas course"
            ]
        ),
        ImprovementSuggestion(
            category=SuggestionCategory.EXPERIENCE,
            priority="medium",
            suggestion="""
            Expand DevOps practices including infrastructure as code (Terraform, CloudFormation)
            and advanced CI/CD pipeline design with testing automation.
            """,
            impact="""
            DevOps skills are essential for senior developers who need to understand
            the full software development lifecycle and deployment strategies.
            """,
            resources=[
                "Terraform documentation and tutorials",
                "AWS CloudFormation best practices",
                "Jenkins and GitHub Actions advanced workflows",
                "Infrastructure as Code patterns and practices"
            ]
        )
    ]

    # Create match result
    match_result = MatchResult(
        score=MatchScore.VERY_GOOD,
        matched_skills=[
            "Python", "Django", "React", "JavaScript", "PostgreSQL",
            "Redis", "AWS", "Docker", "REST APIs", "Git", "CI/CD"
        ],
        relevant_experiences=relevant_experiences,
        recommended_projects=recommended_projects,
        suggestions=suggestions,
        analysis_details=analysis_details,
        job_title="Senior Full Stack Developer",
        company_name="InnovateTech Solutions",
        analyzed_at=datetime.now(),
        confidence_level=0.91
    )

    # Create customized HTML content
    customized_html = {
        'cv': """
        <div class="cv-container">
            <header class="header">
                <h1>John Alexander Smith</h1>
                <div class="contact-info">
                    <p>Senior Full Stack Developer</p>
                    <p>📧 john.smith@email.com | 📱 (555) 123-4567</p>
                    <p>🔗 linkedin.com/in/johnsmith | 💻 github.com/johnsmith</p>
                    <p>📍 San Francisco, CA | 🌐 johnsmith.dev</p>
                </div>
            </header>

            <section class="section">
                <h2>Professional Summary</h2>
                <p>
                    Experienced Senior Full Stack Developer with 8+ years of expertise in building
                    scalable web applications using Python/Django and React. Proven track record
                    of leading technical initiatives, mentoring development teams, and delivering
                    high-performance solutions that serve millions of users. Strong background in
                    cloud infrastructure, microservices architecture, and agile development practices.
                </p>
            </section>

            <section class="section">
                <h2>Core Technical Skills</h2>
                <div class="skills-grid">
                    <div class="skill-category">
                        <h3>Backend Development</h3>
                        <ul>
                            <li>Python (Expert) • Django (Expert) • Flask</li>
                            <li>REST APIs • GraphQL • Microservices</li>
                            <li>PostgreSQL • Redis • MongoDB</li>
                        </ul>
                    </div>
                    <div class="skill-category">
                        <h3>Frontend Development</h3>
                        <ul>
                            <li>React (Advanced) • JavaScript (ES6+) • TypeScript</li>
                            <li>HTML5 • CSS3 • Responsive Design</li>
                            <li>Vue.js • Redux • Webpack</li>
                        </ul>
                    </div>
                    <div class="skill-category">
                        <h3>Cloud & DevOps</h3>
                        <ul>
                            <li>AWS (EC2, S3, RDS, Lambda, ECS)</li>
                            <li>Docker • Kubernetes (Learning)</li>
                            <li>CI/CD • Git • Jenkins</li>
                        </ul>
                    </div>
                </div>
            </section>

            <section class="section">
                <h2>Professional Experience</h2>

                <div class="experience-item">
                    <div class="job-header">
                        <div class="job-title">Senior Python Developer</div>
                        <div class="company-name">TechCorp Industries</div>
                        <div class="date-range">March 2021 - Present (3.5 years)</div>
                    </div>
                    <ul class="achievements">
                        <li>Led development of microservices architecture serving 2M+ daily active users</li>
                        <li>Reduced API response times by 65% through advanced database optimization techniques</li>
                        <li>Mentored team of 6 developers and established comprehensive code review processes</li>
                        <li>Implemented CI/CD pipeline reducing deployment time from hours to minutes</li>
                        <li>Designed fault-tolerant systems with 99.9% uptime across multiple AWS regions</li>
                    </ul>
                </div>

                <div class="experience-item">
                    <div class="job-header">
                        <div class="job-title">Full Stack Engineer</div>
                        <div class="company-name">StartupXYZ</div>
                        <div class="date-range">June 2019 - February 2021 (2.5 years)</div>
                    </div>
                    <ul class="achievements">
                        <li>Built MVP from scratch that secured $5M Series A funding within 18 months</li>
                        <li>Developed real-time collaborative features using WebSockets and Redis</li>
                        <li>Implemented comprehensive testing suite achieving 95% code coverage</li>
                        <li>Designed and deployed scalable database architecture supporting rapid growth</li>
                        <li>Collaborated with product team to define technical requirements and roadmap</li>
                    </ul>
                </div>

                <div class="experience-item">
                    <div class="job-header">
                        <div class="job-title">Backend Developer</div>
                        <div class="company-name">CloudSys Solutions</div>
                        <div class="date-range">September 2017 - May 2019 (2 years)</div>
                    </div>
                    <ul class="achievements">
                        <li>Migrated legacy monolithic application to cloud-native microservices</li>
                        <li>Implemented robust API gateway with rate limiting and comprehensive monitoring</li>
                        <li>Reduced infrastructure costs by 40% through strategic AWS optimization</li>
                        <li>Built automated deployment pipeline using Docker and AWS ECS</li>
                    </ul>
                </div>
            </section>

            <section class="section">
                <h2>Education & Certifications</h2>
                <div class="education-item">
                    <div class="degree">Bachelor of Science in Computer Science</div>
                    <div class="institution">University of California, Berkeley</div>
                    <div class="date-range">2013 - 2017</div>
                    <p>Relevant Coursework: Data Structures, Algorithms, Database Systems, Software Engineering</p>
                </div>
                <div class="certifications">
                    <h3>Professional Certifications</h3>
                    <ul>
                        <li>AWS Certified Solutions Architect - Associate (2022)</li>
                        <li>Certified Kubernetes Application Developer (In Progress)</li>
                    </ul>
                </div>
            </section>

            <section class="section">
                <h2>Key Projects</h2>

                <div class="project-item">
                    <div class="project-title">E-Commerce Platform with Microservices</div>
                    <div class="project-tech">Python • Django • React • PostgreSQL • AWS • Docker</div>
                    <p>
                        Built comprehensive e-commerce platform serving 500K+ active users with microservices
                        architecture. Features include real-time inventory management, payment processing,
                        order tracking, and analytics dashboard. Achieved 99.9% uptime with auto-scaling capabilities.
                    </p>
                </div>

                <div class="project-item">
                    <div class="project-title">Real-Time Collaboration Tool</div>
                    <div class="project-tech">Python • Django • React • WebSockets • Redis</div>
                    <p>
                        Developed real-time collaborative workspace with document editing, video conferencing,
                        and team chat features. Supports 10,000+ concurrent users with conflict resolution
                        and real-time synchronization across multiple time zones.
                    </p>
                </div>

                <div class="project-item">
                    <div class="project-title">API Gateway and Service Mesh</div>
                    <div class="project-tech">Python • Docker • Kubernetes • AWS • Microservices</div>
                    <p>
                        Implemented enterprise-grade API gateway with service discovery, load balancing,
                        authentication, and monitoring. Deployed using Kubernetes with auto-scaling
                        capabilities supporting 1M+ requests per hour.
                    </p>
                </div>
            </section>
        </div>
        """,

        'cover_letter': """
        <div class="letter-container">
            <div class="letter-header">
                <div class="sender-info">
                    <h3>John Alexander Smith</h3>
                    <p>Senior Full Stack Developer</p>
                    <p>📧 john.smith@email.com</p>
                    <p>📱 (555) 123-4567</p>
                    <p>🔗 linkedin.com/in/johnsmith</p>
                </div>
            </div>

            <div class="letter-date">
                <p>December 15, 2024</p>
            </div>

            <div class="letter-address">
                <p>
                    Hiring Manager<br>
                    InnovateTech Solutions<br>
                    Engineering Department<br>
                    456 Innovation Drive<br>
                    San Francisco, CA 94107
                </p>
            </div>

            <div class="letter-greeting">
                <p>Dear InnovateTech Solutions Hiring Team,</p>
            </div>

            <div class="letter-body">
                <p>
                    I am writing to express my strong interest in the Senior Full Stack Developer position
                    at InnovateTech Solutions. With over 8 years of experience building scalable web applications
                    and leading technical initiatives, I am excited about the opportunity to contribute to your
                    innovative engineering team and help drive the next phase of growth for your cutting-edge products.
                </p>

                <p>
                    Your job posting immediately caught my attention because it perfectly aligns with my expertise
                    and passion for full-stack development. In my current role as Senior Python Developer at
                    TechCorp Industries, I have successfully led the development of microservices architecture
                    serving over 2 million daily active users, which directly parallels InnovateTech's scale
                    and technical challenges. My comprehensive experience with Python/Django backend development,
                    React frontend applications, and AWS cloud infrastructure matches your core technology stack
                    requirements perfectly.
                </p>

                <p>
                    What particularly excites me about InnovateTech Solutions is your commitment to building
                    products that serve millions of users worldwide while maintaining a culture of innovation
                    and technical excellence. My track record includes reducing API response times by 65%
                    through strategic optimization, mentoring development teams, and implementing CI/CD pipelines
                    that dramatically improve deployment efficiency. I thrive in collaborative environments
                    where I can combine technical leadership with hands-on development to deliver exceptional results.
                </p>

                <p>
                    Beyond my technical capabilities, I bring strong leadership experience having mentored
                    6 developers and established code review processes that improved code quality and team
                    productivity. My experience building an MVP that secured $5M in Series A funding demonstrates
                    my ability to translate business requirements into technical solutions that drive growth.
                    I am particularly drawn to InnovateTech's focus on microservices architecture and would
                    be excited to contribute to your cloud-native initiatives.
                </p>

                <p>
                    I would welcome the opportunity to discuss how my technical expertise, leadership experience,
                    and passion for building scalable applications can contribute to InnovateTech Solutions'
                    continued success. Thank you for considering my application, and I look forward to the
                    possibility of joining your innovative engineering team.
                </p>
            </div>

            <div class="letter-closing">
                <p>Sincerely,</p>
                <div class="signature-space"></div>
                <p><strong>John Alexander Smith</strong></p>
                <p>Senior Full Stack Developer</p>
            </div>
        </div>
        """
    }

    # Create original content for comparison
    original_html = {
        'cv': """
        <div class="cv-container">
            <header>
                <h1>John Smith</h1>
                <p>Software Developer</p>
                <p>john.smith@email.com | (555) 123-4567</p>
            </header>

            <section>
                <h2>Summary</h2>
                <p>Software developer with experience in web development.</p>
            </section>

            <section>
                <h2>Skills</h2>
                <ul>
                    <li>Python</li>
                    <li>JavaScript</li>
                    <li>SQL</li>
                </ul>
            </section>

            <section>
                <h2>Experience</h2>
                <div>
                    <h3>Developer - TechCorp</h3>
                    <p>Worked on various projects using Python and JavaScript.</p>
                </div>
            </section>
        </div>
        """,

        'cover_letter': """
        <div class="letter-container">
            <p>Dear Hiring Manager,</p>
            <p>I am interested in the software developer position at your company.</p>
            <p>I have experience with Python and web development.</p>
            <p>Thank you for your consideration.</p>
            <p>Sincerely,<br>John Smith</p>
        </div>
        """
    }

    # Create changes list
    changes = [
        "Enhanced professional summary to highlight 8+ years of senior-level experience and technical leadership",
        "Added comprehensive technical skills breakdown organized by category (Backend, Frontend, Cloud & DevOps)",
        "Expanded experience descriptions with quantifiable achievements and metrics (2M+ users, 65% performance improvement)",
        "Included specific technology stack alignment with job requirements (Python/Django, React, AWS)",
        "Added leadership and mentoring experience with team size details (6 developers)",
        "Incorporated relevant project portfolio demonstrating microservices and scalable architecture experience",
        "Enhanced cover letter with company-specific research and value proposition alignment",
        "Emphasized cloud-native development experience and microservices architecture expertise",
        "Added professional certifications and continuous learning initiatives",
        "Tailored achievements to match job posting requirements and company scale (millions of users)",
        "Included specific business impact examples (MVP securing $5M funding, 40% cost reduction)",
        "Structured content for optimal ATS compatibility and visual hierarchy"
    ]

    return {
        'job_data': job_data,
        'match_result': match_result,
        'customized_html': customized_html,
        'original_html': original_html,
        'changes': changes
    }


def demonstrate_preview_generation(demo_data):
    """Demonstrate the preview generation functionality."""
    print("\n" + "="*80)
    print("📋 PREVIEW GENERATION DEMONSTRATION")
    print("="*80)

    try:
        # Initialize preview generator with professional options
        preview_options = PreviewOptions(
            show_side_by_side=True,
            include_match_analysis=True,
            enable_interactive_approval=True,
            highlight_changes=True,
            theme="professional"
        )

        preview_generator = PreviewGenerator(preview_options)

        print("🔄 Generating interactive preview...")

        # Generate the preview
        preview_html = preview_generator.generate_preview(
            customized_html=demo_data['customized_html'],
            match_result=demo_data['match_result'],
            changes=demo_data['changes'],
            original_html=demo_data['original_html']
        )

        # Save preview to file
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        preview_path = output_dir / "integration_demo_preview.html"

        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write(preview_html)

        print(f"✅ Preview generated successfully!")
        print(f"📁 File location: {preview_path}")
        print(f"📊 Match score: {demo_data['match_result'].score}/10 ({demo_data['match_result'].get_match_category()})")
        print(f"🔧 Total customizations: {len(demo_data['changes'])}")
        print(f"🌐 Open the HTML file in a browser to view the interactive preview")

        # Display some key statistics
        match_result = demo_data['match_result']
        print(f"\n📈 Key Statistics:")
        print(f"   • Skills match: {match_result.analysis_details.skills_match_percentage:.1f}%")
        print(f"   • Experience match: {match_result.analysis_details.experience_match_percentage:.1f}%")
        print(f"   • Matched skills: {len(match_result.matched_skills)}")
        print(f"   • Relevant experiences: {len(match_result.relevant_experiences)}")
        print(f"   • Recommended projects: {len(match_result.recommended_projects)}")
        print(f"   • Confidence level: {match_result.confidence_level:.1%}")

        return preview_path

    except Exception as e:
        print(f"❌ Error in preview generation: {e}")
        logger.exception("Preview generation failed")
        return None


def demonstrate_pdf_generation(demo_data):
    """Demonstrate the PDF generation functionality."""
    print("\n" + "="*80)
    print("📄 PDF GENERATION DEMONSTRATION")
    print("="*80)

    try:
        # Configure PDF options for high-quality output
        pdf_options = PDFOptions(
            format="A4",
            margin_top="1.5cm",
            margin_bottom="1.5cm",
            margin_left="2cm",
            margin_right="2cm",
            print_background=True,
            prefer_css_page_size=True,
            generate_tagged_pdf=True,  # For accessibility
            timeout=45000,  # 45 seconds
            scale=0.9,  # Slightly smaller for better fit
            wait_for_timeout=3000  # Wait 3 seconds for rendering
        )

        print("🔄 Generating PDFs with optimized settings...")
        print(f"   📋 Company: {demo_data['job_data'].company_name}")
        print(f"   💼 Position: {demo_data['job_data'].position}")

        # Generate PDFs using the synchronous wrapper
        pdf_paths = generate_application_pdfs_sync(
            cv_html=demo_data['customized_html']['cv'],
            cover_letter_html=demo_data['customized_html']['cover_letter'],
            job_data=demo_data['job_data'],
            output_folder=None,  # Will auto-generate organized folder
            options=pdf_options
        )

        print(f"✅ PDFs generated successfully!")

        # Validate and display information about generated PDFs
        pdf_generator = PDFGenerator()
        total_size = 0

        for doc_type, pdf_path in pdf_paths.items():
            # Validate the PDF
            is_valid = pdf_generator.validate_pdf_output(pdf_path)

            # Get file information
            pdf_file = Path(pdf_path)
            file_size = pdf_file.stat().st_size
            total_size += file_size

            print(f"\n📄 {doc_type.upper()} PDF:")
            print(f"   📁 Path: {pdf_path}")
            print(f"   📊 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"   ✓ Valid: {'YES' if is_valid else 'NO'}")

            # Additional validation details
            if is_valid:
                if file_size > 1024 * 1024:  # > 1MB
                    print(f"   ⚠️  Large file size: {file_size/(1024*1024):.1f} MB")
                else:
                    print(f"   ✅ Optimal file size")

        print(f"\n📊 GENERATION SUMMARY:")
        print(f"   📄 Total PDFs: {len(pdf_paths)}")
        print(f"   💾 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        print(f"   📁 Output folder: {Path(list(pdf_paths.values())[0]).parent.parent}")

        return pdf_paths

    except Exception as e:
        print(f"❌ Error in PDF generation: {e}")
        logger.exception("PDF generation failed")
        return None


async def demonstrate_async_pdf_generation(demo_data):
    """Demonstrate asynchronous PDF generation with advanced features."""
    print("\n" + "="*80)
    print("⚡ ASYNC PDF GENERATION DEMONSTRATION")
    print("="*80)

    try:
        # Configure advanced PDF options
        advanced_options = PDFOptions(
            format="A4",
            margin_top="2cm",
            margin_bottom="2cm",
            margin_left="2cm",
            margin_right="2cm",
            print_background=True,
            prefer_css_page_size=True,
            generate_tagged_pdf=True,
            timeout=60000,  # 60 seconds for complex documents
            wait_for_timeout=5000,  # Wait 5 seconds
            scale=0.85  # Smaller scale for more content per page
        )

        print("🚀 Starting async PDF generation with advanced options...")

        # Use async context manager for proper resource management
        async with PDFGenerator(advanced_options) as generator:
            # Track performance
            start_time = asyncio.get_event_loop().time()

            # Generate PDFs asynchronously
            pdf_paths = await generator.generate_pdfs(
                approved_html=demo_data['customized_html'],
                job_data=demo_data['job_data']
            )

            end_time = asyncio.get_event_loop().time()
            generation_time = end_time - start_time

            print(f"✅ Async PDF generation completed in {generation_time:.2f} seconds!")

            # Validate all generated PDFs
            validation_results = {}
            for doc_type, pdf_path in pdf_paths.items():
                is_valid = generator.validate_pdf_output(pdf_path)
                file_size = Path(pdf_path).stat().st_size

                validation_results[doc_type] = {
                    'valid': is_valid,
                    'size': file_size,
                    'path': pdf_path
                }

                print(f"\n⚡ {doc_type.upper()} (Async):")
                print(f"   📁 {Path(pdf_path).name}")
                print(f"   ✓ Valid: {'YES' if is_valid else 'NO'}")
                print(f"   📊 Size: {file_size:,} bytes")

        print(f"\n⚡ ASYNC GENERATION SUMMARY:")
        print(f"   ⏱️  Generation time: {generation_time:.2f} seconds")
        print(f"   📄 PDFs generated: {len(pdf_paths)}")
        print(f"   🧹 Resources automatically cleaned up")

        return pdf_paths, validation_results

    except Exception as e:
        print(f"❌ Error in async PDF generation: {e}")
        logger.exception("Async PDF generation failed")
        return None, None


def demonstrate_error_handling():
    """Demonstrate comprehensive error handling scenarios."""
    print("\n" + "="*80)
    print("🛡️  ERROR HANDLING DEMONSTRATION")
    print("="*80)

    # Test 1: Preview Generator Validation Errors
    print("\n🧪 Test 1: Preview Generator Input Validation")
    try:
        generator = PreviewGenerator()

        # Invalid HTML structure
        invalid_html = {'invalid_key': 'content'}
        fake_match = create_demo_data()['match_result']

        generator.generate_preview(invalid_html, fake_match, [])

    except Exception as e:
        print(f"✅ Caught expected validation error: {type(e).__name__}: {e}")

    # Test 2: Empty Content Handling
    print("\n🧪 Test 2: Empty Content Handling")
    try:
        generator = PreviewGenerator()
        empty_html = {'cv': '', 'cover_letter': ''}
        fake_match = create_demo_data()['match_result']

        preview = generator.generate_preview(empty_html, fake_match, [])
        print("✅ Successfully handled empty content (generated placeholder)")

    except Exception as e:
        print(f"⚠️  Unexpected error with empty content: {e}")

    # Test 3: PDF Generator Validation
    print("\n🧪 Test 3: PDF Generator Input Validation")
    try:
        # Test without Playwright availability (mocked)
        with patch('src.agents.pdf_generator.PLAYWRIGHT_AVAILABLE', False):
            from src.agents.pdf_generator import PDFGenerator
            PDFGenerator()

    except Exception as e:
        print(f"✅ Caught expected Playwright error: {type(e).__name__}")

    # Test 4: File Validation
    print("\n🧪 Test 4: PDF File Validation")
    try:
        from src.agents.pdf_generator import PDFGenerator
        generator = PDFGenerator()

        # Test with non-existent file
        result = generator.validate_pdf_output("/nonexistent/path/test.pdf")
        print(f"✅ Validation correctly failed for non-existent file: {result}")

    except Exception as e:
        print(f"⚠️  Error in file validation: {e}")

    print("\n🛡️  Error handling tests completed!")


def demonstrate_complete_workflow():
    """Demonstrate the complete end-to-end workflow."""
    print("\n" + "="*80)
    print("🔄 COMPLETE WORKFLOW DEMONSTRATION")
    print("="*80)

    try:
        # Step 1: Create demo data
        print("📋 Step 1: Preparing comprehensive application data...")
        demo_data = create_demo_data()

        print(f"   🏢 Company: {demo_data['job_data'].company_name}")
        print(f"   💼 Position: {demo_data['job_data'].position}")
        print(f"   📊 Match Score: {demo_data['match_result'].score}/10")
        print(f"   🔧 Customizations: {len(demo_data['changes'])}")

        # Step 2: Generate preview
        print("\n👀 Step 2: Generating interactive preview...")
        preview_path = demonstrate_preview_generation(demo_data)

        if preview_path:
            print(f"   ✅ Preview available for user review")

            # Step 3: Simulate user approval
            print("\n👤 Step 3: Simulating user approval process...")
            print("   🤔 User reviewing application materials...")
            print("   ✅ User approved the customized application!")

            # Step 4: Generate PDFs
            print("\n📄 Step 4: Generating final PDF documents...")
            pdf_paths = demonstrate_pdf_generation(demo_data)

            if pdf_paths:
                # Step 5: Async generation for comparison
                print("\n⚡ Step 5: Demonstrating async PDF generation...")
                async_paths, validation_results = await demonstrate_async_pdf_generation(demo_data)

                # Step 6: Final summary
                print("\n📊 COMPLETE WORKFLOW SUMMARY:")
                print("="*50)
                print(f"🏢 Company: {demo_data['job_data'].company_name}")
                print(f"💼 Position: {demo_data['job_data'].position}")
                print(f"📈 Match Score: {demo_data['match_result'].score}/10 ({demo_data['match_result'].get_match_category()})")
                print(f"🎯 Skills Match: {demo_data['match_result'].analysis_details.skills_match_percentage:.1f}%")
                print(f"💼 Experience Match: {demo_data['match_result'].analysis_details.experience_match_percentage:.1f}%")
                print(f"🔧 Total Customizations: {len(demo_data['changes'])}")
                print(f"👀 Interactive Preview: Generated")
                print(f"📄 PDF Documents: {len(pdf_paths) if pdf_paths else 0} generated")
                print(f"⚡ Async Generation: {'Successful' if async_paths else 'Failed'}")
                print(f"✅ Workflow Status: COMPLETED SUCCESSFULLY")

                return True
            else:
                print("❌ PDF generation failed - workflow incomplete")
                return False
        else:
            print("❌ Preview generation failed - workflow stopped")
            return False

    except Exception as e:
        print(f"❌ Complete workflow failed: {e}")
        logger.exception("Complete workflow demonstration failed")
        return False


async def main():
    """Main integration demo function."""
    print("🚀 PREVIEW AND PDF GENERATION SYSTEM - INTEGRATION DEMO")
    print("="*80)
    print("This demo showcases the complete functionality of the preview and PDF generation system.")
    print("Features demonstrated:")
    print("  • Interactive HTML preview generation with match analysis")
    print("  • Side-by-side content comparison")
    print("  • Professional PDF generation with Playwright")
    print("  • Comprehensive error handling and validation")
    print("  • Async/sync operation modes")
    print("  • File organization and naming conventions")

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    try:
        # Run individual demonstrations
        demo_data = create_demo_data()

        # 1. Preview generation
        demonstrate_preview_generation(demo_data)

        # 2. PDF generation
        demonstrate_pdf_generation(demo_data)

        # 3. Error handling
        demonstrate_error_handling()

        # 4. Complete workflow
        success = await demonstrate_complete_workflow()

        # Final summary
        print("\n" + "="*80)
        if success:
            print("🎉 INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
            print("📁 Check the 'output' directory for generated files:")
            print("   • integration_demo_preview.html - Interactive preview")
            print("   • applications/[company_date]/ - PDF documents and metadata")
            print("\n🌐 Open the HTML preview in a browser to see the interactive features")
            print("📄 PDF files can be opened with any PDF viewer")
        else:
            print("⚠️  INTEGRATION DEMO COMPLETED WITH SOME ISSUES")
            print("Check the output above for specific error details")

        print("\n📚 For more information, see:")
        print("   • docs/PREVIEW_PDF_SYSTEM.md - Complete documentation")
        print("   • docs/PLAYWRIGHT_SETUP.md - Browser setup guide")
        print("   • examples/preview_and_pdf_examples.py - Additional examples")

    except Exception as e:
        print(f"\n❌ Integration demo failed: {e}")
        logger.exception("Integration demo failed")


if __name__ == "__main__":
    # Ensure proper async handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        logger.exception("Demo crashed")