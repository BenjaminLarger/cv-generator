#!/usr/bin/env python3
"""
CV Generator - Main entry point for the application.

This module provides the main entry point for the CV/Cover Letter generator.
It sets up logging, configuration, and provides a basic CLI interface.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging_config import setup_logging, get_logger


def setup_cli() -> argparse.ArgumentParser:
    """Set up command line interface."""
    parser = argparse.ArgumentParser(
        description="CV/Cover Letter Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --help                 Show this help message
  python main.py --version              Show version information
  python main.py --debug                Run in debug mode

Environment Variables:
  OPENAI_API_KEY                        Required: Your OpenAI API key
  CV_GEN_LOG_LEVEL                      Optional: Logging level (DEBUG, INFO, WARNING, ERROR)
  CV_GEN_DEBUG                          Optional: Enable debug mode (true/false)
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="CV Generator v0.1.0"
    )

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

    return parser


def main():
    """Main application entry point."""
    parser = setup_cli()
    args = parser.parse_args()

    # Set up logging
    log_level = args.log_level or ("DEBUG" if args.debug else None)
    logger = setup_logging(log_level=log_level)

    logger.info("Starting CV Generator application")

    try:
        # Import configuration after logging is set up
        from config.config import config

        logger.info(f"Configuration loaded successfully")
        logger.info(f"Project root: {config.project_root}")
        logger.info(f"Templates directory: {config.templates_dir}")
        logger.info(f"Applications directory: {config.applications_dir}")
        logger.info(f"Logs directory: {config.logs_dir}")

        if args.debug:
            logger.debug("Debug mode enabled")
            logger.debug(f"OpenAI model: {config.openai.model}")
            logger.debug(f"Temperature: {config.openai.temperature}")
            logger.debug(f"Max tokens: {config.openai.max_tokens}")

        # TODO: Add main application logic here
        logger.info("CV Generator is ready! (Main functionality to be implemented)")

        # For now, just show available directories
        print("\nCV Generator v0.1.0")
        print("===================")
        print("\nProject structure:")
        print(f"  Templates: {config.templates_dir}")
        print(f"  Applications: {config.applications_dir}")
        print(f"  Logs: {config.logs_dir}")
        print("\nConfiguration:")
        print(f"  OpenAI Model: {config.openai.model}")
        print(f"  Log Level: {config.log_level}")
        print(f"  Debug Mode: {config.debug}")

        print("\nNext steps:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Create CV templates in the templates/ directory")
        print("3. Implement agent logic in src/agents/")
        print("4. Define data models in src/models/")

    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        if args.debug:
            logger.exception("Full error details:")
        sys.exit(1)

    logger.info("Application completed successfully")


if __name__ == "__main__":
    main()