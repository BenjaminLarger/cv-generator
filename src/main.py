#!/usr/bin/env python3
"""
CV Generator - Main entry point for the comprehensive workflow application.

This module provides the main entry point for the CV/Cover Letter generator
with complete LangGraph workflow orchestration, human-in-the-loop processes,
and comprehensive CLI interface supporting both interactive and batch modes.
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging_config import setup_logging, get_logger
from workflows.cv_generation_workflow import CVGenerationWorkflow
from workflows.state_manager import StateManager
from config.config import config


def setup_cli() -> argparse.ArgumentParser:
    """Set up comprehensive command line interface."""
    parser = argparse.ArgumentParser(
        description="CV/Cover Letter Generator with LangGraph Workflow Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with job URL
  python main.py --job-url "https://example.com/job" --profile user_profile.yaml

  # Batch mode with job text file
  python main.py --job-text "$(cat job.txt)" --profile user_profile.yaml --batch

  # Resume interrupted workflow
  python main.py --resume workflow_state.json

  # Check workflow status
  python main.py --status workflow-id-12345

Environment Variables:
  OPENAI_API_KEY                        Required: Your OpenAI API key
  CV_GEN_LOG_LEVEL                      Optional: Logging level (DEBUG, INFO, WARNING, ERROR)
  CV_GEN_DEBUG                          Optional: Enable debug mode (true/false)
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="CV Generator v1.0.0 with LangGraph Workflow"
    )

    # Job input options (mutually exclusive)
    job_group = parser.add_mutually_exclusive_group()
    job_group.add_argument(
        "--job-url",
        type=str,
        help="URL to the job posting"
    )
    job_group.add_argument(
        "--job-text",
        type=str,
        help="Raw job posting text (or path to text file starting with @)"
    )

    # Profile and output options
    parser.add_argument(
        "--profile",
        type=str,
        required=False,
        help="Path to user profile YAML file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./applications",
        help="Output directory for generated files (default: ./applications)"
    )

    # Workflow mode options
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run in interactive mode with human approval steps (default)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run in batch mode without human interaction"
    )

    # Resume and status options
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume workflow from checkpoint file"
    )

    parser.add_argument(
        "--status",
        type=str,
        help="Check status of workflow by ID"
    )

    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )

    parser.add_argument(
        "--state-dir",
        type=str,
        default="./workflow_states",
        help="Directory for workflow state files (default: ./workflow_states)"
    )

    # Debugging and logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (overrides environment variable)"
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up temporary files (useful for debugging)"
    )

    return parser


def validate_environment() -> bool:
    """Validate that required environment variables and dependencies are available."""
    logger = get_logger(__name__)

    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        print("\n❌ ERROR: OpenAI API key not found")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return False

    # Check for required directories
    try:
        config.templates_dir.mkdir(parents=True, exist_ok=True)
        config.applications_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create required directories: {e}")
        return False

    return True


def load_job_input(job_arg: str) -> str:
    """Load job input from argument (file path or direct text)."""
    if job_arg.startswith('@'):
        # Load from file
        file_path = Path(job_arg[1:])
        if not file_path.exists():
            raise FileNotFoundError(f"Job text file not found: {file_path}")
        return file_path.read_text(encoding='utf-8')
    else:
        # Direct text
        return job_arg


async def run_interactive_mode(
    workflow: CVGenerationWorkflow,
    job_input: str,
    profile_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Run workflow in interactive mode with user guidance."""
    logger = get_logger(__name__)

    print("\n🚀 Starting CV Generation Workflow (Interactive Mode)")
    print("=" * 60)

    # Show job and profile info
    print(f"📋 Job Input: {job_input[:100]}{'...' if len(job_input) > 100 else ''}")
    print(f"👤 Profile: {profile_path}")
    print(f"📁 Output: {output_dir}")

    # Confirm to proceed
    proceed = input("\nProceed with workflow? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Workflow cancelled by user.")
        sys.exit(0)

    # Run workflow
    try:
        result = await workflow.run_workflow(
            job_input=job_input,
            user_profile_path=profile_path,
            output_directory=output_dir,
            interactive_mode=True
        )

        # Show results
        print("\n✅ Workflow Completed Successfully!")
        print("=" * 40)
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")
        print(f"Progress: {result['progress']:.1f}%")

        if result.get('output_files'):
            print("\n📄 Generated Files:")
            for file_type, file_path in result['output_files'].items():
                print(f"  {file_type}: {file_path}")

        if result.get('match_score'):
            print(f"\n🎯 Match Score: {result['match_score']}/10")

        if result.get('errors'):
            print(f"\n⚠️  Warnings/Errors: {len(result['errors'])}")
            for error in result['errors'][:3]:  # Show first 3
                print(f"  - {error}")

        return result

    except Exception as e:
        logger.error(f"Interactive workflow failed: {e}")
        print(f"\n❌ Workflow failed: {e}")
        raise


async def run_batch_mode(
    workflow: CVGenerationWorkflow,
    job_input: str,
    profile_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """Run workflow in batch mode without user interaction."""
    logger = get_logger(__name__)

    print("\n🔄 Starting CV Generation Workflow (Batch Mode)")
    print("=" * 50)

    try:
        result = await workflow.run_workflow(
            job_input=job_input,
            user_profile_path=profile_path,
            output_directory=output_dir,
            interactive_mode=False
        )

        # Show brief results
        status_symbol = "✅" if result['status'] == 'completed' else "❌"
        print(f"\n{status_symbol} Batch processing {result['status']}")
        print(f"Duration: {result['duration_seconds']:.1f}s | Progress: {result['progress']:.1f}%")

        if result.get('output_files'):
            print(f"Generated {len(result['output_files'])} files in {output_dir}")

        return result

    except Exception as e:
        logger.error(f"Batch workflow failed: {e}")
        print(f"\n❌ Batch processing failed: {e}")
        raise


def show_workflow_status(state_manager: StateManager, workflow_id: str) -> None:
    """Show workflow status information."""
    try:
        checkpoints = state_manager.list_checkpoints(workflow_id)

        if not checkpoints:
            print(f"\n❌ No workflow found with ID: {workflow_id}")
            return

        latest = checkpoints[0]
        print(f"\n📊 Workflow Status: {workflow_id}")
        print("=" * 50)
        print(f"Current Step: {latest.step_name}")
        print(f"Last Update: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Progress: {latest.metadata.get('progress', 0):.1f}%")
        print(f"Status: {latest.metadata.get('status', 'unknown')}")
        print(f"Total Checkpoints: {len(checkpoints)}")

        if latest.metadata.get('errors_count', 0) > 0:
            print(f"❌ Errors: {latest.metadata['errors_count']}")

    except Exception as e:
        print(f"\n❌ Failed to get workflow status: {e}")


async def main():
    """Main application entry point with complete workflow orchestration."""
    parser = setup_cli()
    args = parser.parse_args()

    # Set up logging
    log_level = args.log_level or ("DEBUG" if args.debug else None)
    logger = setup_logging(log_level=log_level)

    logger.info("Starting CV Generator with LangGraph Workflow")

    try:
        # Validate environment
        if not validate_environment():
            sys.exit(1)

        # Initialize state manager
        state_manager = StateManager(
            state_directory=args.state_dir,
            compress_states=True,
            auto_cleanup=not args.no_cleanup
        )

        # Handle status check
        if args.status:
            show_workflow_status(state_manager, args.status)
            return

        # Handle resume workflow
        if args.resume:
            print(f"\n🔄 Resuming workflow from: {args.resume}")
            try:
                state = state_manager.load_state(args.resume)
                workflow = CVGenerationWorkflow(
                    config=config.__dict__,
                    state_manager=state_manager,
                    debug_mode=args.debug
                )

                # Resume with current interactive setting
                interactive_mode = not args.batch
                result = await workflow.run_workflow(
                    job_input=state['job_input'],
                    user_profile_path=state['user_profile_path'],
                    output_directory=state['output_directory'],
                    interactive_mode=interactive_mode,
                    resume_from_checkpoint=args.resume
                )

                print(f"\n✅ Resumed workflow completed: {result['status']}")
                return

            except Exception as e:
                logger.error(f"Failed to resume workflow: {e}")
                print(f"\n❌ Failed to resume workflow: {e}")
                sys.exit(1)

        # Validate required arguments for new workflow
        if not (args.job_url or args.job_text):
            print("\n❌ ERROR: Either --job-url or --job-text is required")
            parser.print_help()
            sys.exit(1)

        if not args.profile:
            print("\n❌ ERROR: --profile is required")
            parser.print_help()
            sys.exit(1)

        # Load job input
        try:
            job_input = args.job_url or load_job_input(args.job_text)
        except Exception as e:
            logger.error(f"Failed to load job input: {e}")
            print(f"\n❌ Failed to load job input: {e}")
            sys.exit(1)

        # Validate profile file
        profile_path = Path(args.profile)
        if not profile_path.exists():
            print(f"\n❌ Profile file not found: {profile_path}")
            sys.exit(1)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize workflow
        workflow_config = {
            'openai_model': config.openai.model,
            'max_retries': 3,
            'templates_dir': config.templates_dir,
            'output_dir': output_dir
        }

        workflow = CVGenerationWorkflow(
            config=workflow_config,
            state_manager=state_manager,
            debug_mode=args.debug
        )

        # Run workflow based on mode
        if args.batch:
            result = await run_batch_mode(
                workflow, job_input, str(profile_path), str(output_dir)
            )
        else:
            result = await run_interactive_mode(
                workflow, job_input, str(profile_path), str(output_dir)
            )

        # Show final summary
        print(f"\n📋 Workflow Summary")
        print(f"ID: {result['workflow_id']}")
        print(f"Status: {result['status']}")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")

        if result['status'] == 'completed':
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Application failed: {e}")
        if args.debug:
            logger.exception("Full error details:")
        print(f"\n❌ Application failed: {e}")
        sys.exit(1)


def run_main():
    """Synchronous wrapper for async main function."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    run_main()