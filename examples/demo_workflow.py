#!/usr/bin/env python3
"""
Demo script for CV Generation Workflow.

This script demonstrates the complete LangGraph workflow system for CV/Cover Letter
generation, including interactive and batch modes, state management, and error recovery.

Usage:
    python examples/demo_workflow.py --interactive
    python examples/demo_workflow.py --batch
    python examples/demo_workflow.py --resume <checkpoint_file>
"""

import asyncio
import os
import sys
from pathlib import Path
import argparse
import tempfile
from typing import Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from workflows.cv_generation_workflow import CVGenerationWorkflow
from workflows.state_manager import StateManager
from utils.logging_config import setup_logging
from config.config import config


class WorkflowDemo:
    """Demo class for showcasing workflow capabilities."""

    def __init__(self, demo_dir: Path):
        """Initialize demo environment."""
        self.demo_dir = demo_dir
        self.setup_demo_environment()

    def setup_demo_environment(self) -> None:
        """Set up demo files and directories."""
        # Create demo directories
        self.state_dir = self.demo_dir / "workflow_states"
        self.output_dir = self.demo_dir / "generated_applications"
        self.templates_dir = self.demo_dir / "templates"

        for directory in [self.state_dir, self.output_dir, self.templates_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Copy sample profile
        sample_profile_src = project_root / "examples" / "sample_user_profile.yaml"
        self.profile_path = self.demo_dir / "demo_profile.yaml"

        if sample_profile_src.exists():
            import shutil
            shutil.copy2(sample_profile_src, self.profile_path)
        else:
            self.create_minimal_profile()

        # Create sample job descriptions
        self.create_sample_jobs()

        # Copy template files from main templates directory
        self.copy_templates()

        print(f"✅ Demo environment set up in: {self.demo_dir}")
        print(f"📁 Profile: {self.profile_path}")
        print(f"📁 Output: {self.output_dir}")
        print(f"📁 State: {self.state_dir}")

    def create_minimal_profile(self) -> None:
        """Create a minimal user profile for demo."""
        profile_content = '''
personal_info:
  name: "Demo User"
  email: "demo@example.com"
  location: "San Francisco, CA"

skills:
  - "Python"
  - "JavaScript"
  - "React"
  - "Node.js"
  - "SQL"

experiences:
  - company: "Demo Corp"
    role: "Software Developer"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    achievements:
      - "Built web applications using modern frameworks"
      - "Collaborated with cross-functional teams"
    technologies:
      - "Python"
      - "JavaScript"
      - "React"

education:
  - degree: "Bachelor of Science in Computer Science"
    institution: "Demo University"
    graduation_year: 2019
'''
        self.profile_path.write_text(profile_content.strip())

    def create_sample_jobs(self) -> None:
        """Create sample job descriptions for demo."""
        self.job_descriptions = {
            "senior_engineer": """
Senior Software Engineer - TechFlow Inc.

We are seeking a Senior Software Engineer to join our dynamic team in San Francisco.

Requirements:
- Bachelor's degree in Computer Science or related field
- 5+ years of software development experience
- Strong proficiency in Python, JavaScript, and React
- Experience with cloud platforms (AWS, Azure, or GCP)
- Knowledge of microservices architecture
- Excellent problem-solving and communication skills

Responsibilities:
- Design and develop scalable web applications
- Lead technical discussions and code reviews
- Mentor junior developers
- Collaborate with product and design teams
- Optimize application performance and reliability

We offer competitive salary, equity, and comprehensive benefits.
""",

            "full_stack_developer": """
Full Stack Developer - StartupXYZ

Join our innovative startup as a Full Stack Developer and help build the next generation of web applications.

Required Skills:
- 3+ years of full-stack development experience
- Proficiency in JavaScript, Node.js, and React
- Experience with databases (PostgreSQL, MongoDB)
- Knowledge of RESTful API design
- Familiarity with Git and agile development

Nice to Have:
- Experience with TypeScript
- Knowledge of Docker and containerization
- Understanding of CI/CD practices
- Previous startup experience

What You'll Do:
- Build responsive web applications
- Develop and maintain APIs
- Work closely with designers and product managers
- Participate in architectural decisions
- Contribute to a positive team culture

Remote-friendly position with flexible hours.
""",

            "data_scientist": """
Data Scientist - Analytics Corp

We're looking for a Data Scientist to help drive data-informed decisions across our organization.

Requirements:
- Master's degree in Data Science, Statistics, or related field
- 3+ years of experience in data analysis and machine learning
- Strong programming skills in Python or R
- Experience with SQL and database systems
- Knowledge of statistical modeling and hypothesis testing
- Excellent communication and visualization skills

Technical Skills:
- Python (pandas, scikit-learn, matplotlib)
- SQL and database management
- Machine learning algorithms
- Data visualization tools
- Statistical analysis

Responsibilities:
- Analyze large datasets to identify trends and insights
- Build predictive models and algorithms
- Create data visualizations and reports
- Collaborate with stakeholders to define metrics
- Present findings to executive leadership

Competitive compensation package with growth opportunities.
"""
        }

        # Save job descriptions to files
        for job_name, description in self.job_descriptions.items():
            job_file = self.demo_dir / f"{job_name}_job.txt"
            job_file.write_text(description.strip())

        print(f"📄 Created {len(self.job_descriptions)} sample job descriptions")

    def copy_templates(self) -> None:
        """Copy template files from main templates directory to demo workspace."""
        import shutil

        # Main templates directory
        main_templates_dir = self.demo_dir.parent / "templates"

        if not main_templates_dir.exists():
            # If templates directory doesn't exist, create basic templates
            self.create_basic_templates()
            return

        # Copy all template files
        for template_file in main_templates_dir.glob("*.html"):
            target_file = self.templates_dir / template_file.name
            shutil.copy2(template_file, target_file)

        print(f"📄 Copied templates from {main_templates_dir} to {self.templates_dir}")

    def create_basic_templates(self) -> None:
        """Create basic HTML templates for demo if main templates don't exist."""
        # Basic CV template
        cv_template = """<!DOCTYPE html>
<html>
<head>
    <title><!-- FULL_NAME --> - CV</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 25px; }
        .section-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1><!-- FULL_NAME --></h1>
        <p><!-- EMAIL --> | <!-- PHONE --> | <!-- LOCATION --></p>
    </div>

    <div class="section">
        <div class="section-title">Professional Summary</div>
        <p><!-- PROFESSIONAL_SUMMARY --></p>
    </div>

    <div class="section">
        <div class="section-title">Technical Skills</div>
        <!-- TECHNICAL_SKILLS -->
    </div>

    <div class="section">
        <div class="section-title">Work Experience</div>
        <!-- WORK_EXPERIENCE -->
    </div>
</body>
</html>"""

        cv_template_file = self.templates_dir / "cv_template.html"
        cv_template_file.write_text(cv_template)

        # Basic Cover Letter template
        cover_letter_template = """<!DOCTYPE html>
<html>
<head>
    <title>Cover Letter - <!-- FULL_NAME --></title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
    </style>
</head>
<body>
    <div>
        <p><!-- FULL_NAME --><br>
        <!-- EMAIL --><br>
        <!-- PHONE --></p>

        <p><!-- HIRING_MANAGER --><br>
        <!-- COMPANY_NAME --></p>

        <p>Dear <!-- HIRING_MANAGER -->,</p>

        <p><!-- COVER_LETTER_CONTENT --></p>

        <p>Sincerely,<br>
        <!-- FULL_NAME --></p>
    </div>
</body>
</html>"""

        cover_letter_file = self.templates_dir / "cover_letter_template.html"
        cover_letter_file.write_text(cover_letter_template)

        print(f"📄 Created basic templates in {self.templates_dir}")

    async def run_interactive_demo(self) -> Dict[str, Any]:
        """Run interactive workflow demo."""
        print("\n🎯 Starting Interactive Workflow Demo")
        print("=" * 50)

        # Show available job descriptions
        print("\nAvailable job descriptions:")
        for i, (job_name, _) in enumerate(self.job_descriptions.items(), 1):
            print(f"  {i}. {job_name.replace('_', ' ').title()}")

        while True:
            try:
                choice = input(f"\nSelect job (1-{len(self.job_descriptions)}): ").strip()
                job_index = int(choice) - 1
                job_name = list(self.job_descriptions.keys())[job_index]
                job_text = self.job_descriptions[job_name]
                break
            except (ValueError, IndexError):
                print("Invalid choice. Please try again.")

        print(f"\n✅ Selected: {job_name.replace('_', ' ').title()}")

        # Initialize workflow
        state_manager = StateManager(state_directory=self.state_dir)
        workflow_config = {
            'openai_model': 'gpt-4',
            'max_retries': 3,
            'templates_dir': self.templates_dir,
            'output_dir': self.output_dir
        }

        workflow = CVGenerationWorkflow(
            config=workflow_config,
            state_manager=state_manager,
            debug_mode=True
        )

        try:
            result = await workflow.run_workflow(
                job_input=job_text,
                user_profile_path=str(self.profile_path),
                output_directory=str(self.output_dir),
                interactive_mode=True
            )

            print("\n🎉 Interactive Demo Completed!")
            print("=" * 40)
            self.print_results(result)
            return result

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            raise

    async def run_batch_demo(self) -> Dict[str, Any]:
        """Run batch workflow demo."""
        print("\n⚡ Starting Batch Workflow Demo")
        print("=" * 40)

        # Use first job description for batch demo
        job_name = list(self.job_descriptions.keys())[0]
        job_text = self.job_descriptions[job_name]

        print(f"📋 Processing: {job_name.replace('_', ' ').title()}")
        print(f"👤 Profile: {self.profile_path.name}")

        # Initialize workflow
        state_manager = StateManager(state_directory=self.state_dir)
        workflow_config = {
            'openai_model': 'gpt-4',
            'max_retries': 3,
            'templates_dir': self.templates_dir,
            'output_dir': self.output_dir
        }

        workflow = CVGenerationWorkflow(
            config=workflow_config,
            state_manager=state_manager,
            debug_mode=False  # Less verbose for batch
        )

        try:
            result = await workflow.run_workflow(
                job_input=job_text,
                user_profile_path=str(self.profile_path),
                output_directory=str(self.output_dir),
                interactive_mode=False
            )

            print("\n🚀 Batch Demo Completed!")
            print("=" * 30)
            self.print_results(result)
            return result

        except Exception as e:
            print(f"\n❌ Batch demo failed: {e}")
            raise

    async def demonstrate_resume_functionality(self) -> None:
        """Demonstrate workflow resume from checkpoint."""
        print("\n🔄 Demonstrating Resume Functionality")
        print("=" * 45)

        # List available checkpoints
        state_manager = StateManager(state_directory=self.state_dir)
        checkpoints = state_manager.list_checkpoints()

        if not checkpoints:
            print("No checkpoints available. Run a workflow first.")
            return

        print(f"Found {len(checkpoints)} checkpoints:")
        for i, checkpoint in enumerate(checkpoints[:5], 1):  # Show first 5
            print(f"  {i}. {checkpoint.step_name} - {checkpoint.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # For demo, just show the functionality
        latest = checkpoints[0]
        print(f"\n📁 Latest checkpoint: {latest.step_name}")
        print(f"🕒 Timestamp: {latest.timestamp}")
        print(f"📊 Progress: {latest.metadata.get('progress', 0):.1f}%")

        print("\n💡 To resume a workflow, use:")
        print(f"   python src/main.py --resume <checkpoint_file>")

    def print_results(self, result: Dict[str, Any]) -> None:
        """Print workflow results in a formatted way."""
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")
        print(f"Progress: {result['progress']:.1f}%")

        if result.get('match_score'):
            print(f"Match Score: {result['match_score']}/10")

        if result.get('output_files'):
            print(f"\n📄 Generated Files:")
            for file_type, file_path in result['output_files'].items():
                print(f"  {file_type}: {file_path}")

        if result.get('errors'):
            print(f"\n⚠️  Warnings/Errors: {len(result['errors'])}")
            for error in result['errors'][:3]:  # Show first 3
                print(f"  - {error}")

    def show_demo_menu(self) -> str:
        """Show demo menu and get user choice."""
        print("\n🎮 CV Generator Workflow Demo")
        print("=" * 35)
        print("1. Interactive Mode Demo")
        print("2. Batch Mode Demo")
        print("3. Show Resume Functionality")
        print("4. View Generated Files")
        print("5. Clean Demo Environment")
        print("6. Exit")

        while True:
            choice = input("\nSelect option (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            print("Invalid choice. Please select 1-6.")

    def view_generated_files(self) -> None:
        """Show generated files in output directory."""
        print("\n📁 Generated Files")
        print("=" * 20)

        files = list(self.output_dir.rglob("*"))
        if not files:
            print("No files generated yet. Run a workflow first.")
            return

        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"  📄 {file_path.name} ({size} bytes)")

    def clean_demo_environment(self) -> None:
        """Clean up demo environment."""
        import shutil

        print("\n🧹 Cleaning Demo Environment")
        print("=" * 30)

        try:
            shutil.rmtree(self.output_dir)
            shutil.rmtree(self.state_dir)
            self.output_dir.mkdir(exist_ok=True)
            self.state_dir.mkdir(exist_ok=True)
            print("✅ Demo environment cleaned successfully")
        except Exception as e:
            print(f"❌ Failed to clean environment: {e}")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="CV Generator Workflow Demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--batch", action="store_true", help="Run batch demo")
    parser.add_argument("--resume", action="store_true", help="Show resume functionality")
    parser.add_argument("--demo-dir", type=str, help="Demo directory path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(log_level=log_level)

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Set up demo directory
    if args.demo_dir:
        demo_dir = Path(args.demo_dir)
    else:
        demo_dir = Path.cwd() / "demo_workspace"

    demo_dir.mkdir(exist_ok=True)

    try:
        demo = WorkflowDemo(demo_dir)

        if args.interactive:
            await demo.run_interactive_demo()
        elif args.batch:
            await demo.run_batch_demo()
        elif args.resume:
            await demo.demonstrate_resume_functionality()
        else:
            # Interactive menu
            while True:
                choice = demo.show_demo_menu()

                if choice == '1':
                    await demo.run_interactive_demo()
                elif choice == '2':
                    await demo.run_batch_demo()
                elif choice == '3':
                    await demo.demonstrate_resume_functionality()
                elif choice == '4':
                    demo.view_generated_files()
                elif choice == '5':
                    demo.clean_demo_environment()
                elif choice == '6':
                    print("\n👋 Thanks for trying the CV Generator Workflow Demo!")
                    break

    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())