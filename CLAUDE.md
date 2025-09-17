# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered CV/Cover Letter generator using LangGraph workflows, OpenAI GPT models, and Python 3.12. Built with agent-based architecture for job analysis, profile matching, and document generation.

## Essential Commands

```bash
# Development environment
source .venv/bin/activate                    # Activate virtual environment
uv pip install -r requirements.txt          # Install dependencies
playwright install                          # Install browser binaries

# Testing
uv run pytest                              # Run all tests
uv run pytest tests/test_specific.py       # Run specific test file
uv run pytest -v                           # Verbose test output
uv run pytest -k "test_pattern"            # Run tests matching pattern

# Application
cd src && python main.py                   # Run main application
cd src && python main.py --debug           # Run with debug logging
python -m src.agents.job_analyzer          # Run specific agent
```

## Architecture

**Agent-Based Workflow**: LangGraph orchestrates specialized AI agents (JobAnalyzer, ProfileMatcher, etc.) that communicate via Pydantic models.

**Data Flow**: JobData → ProfileMatcher → MatchResult → TemplateCustomizer → HTML/PDF output

**Core Models**:
- `JobData`: Structured job posting data with validation
- `UserProfile`: Complete user information with nested models (PersonalInfo, WorkExperience, etc.)
- `MatchResult`: AI-generated matching analysis with scores and recommendations

**Configuration**: Environment-driven config via Pydantic models in `config/config.py`. Uses `CV_GEN_` prefix for env vars.

**Logging**: Component-specific logging with LogComponent enum (SCRAPING, AGENTS, ERRORS, MAIN) writing to separate log files.

## Key Integration Points

**OpenAI Integration**: All agents use structured GPT prompts with retry logic and error handling. Configure via `OPENAI_API_KEY` environment variable.

**Template System**: HTML templates use comment-based placeholders (`<!-- PLACEHOLDER_NAME -->`) processed by `template_engine.py`.

**Web Scraping**: Robust scraping with user-agent rotation, exponential backoff, and domain validation in `scraper.py`.

**File Management**: Organized output structure in `applications/` with company/date-based folders managed by `file_manager.py`.

## Environment Setup

Required environment variables:
```bash
OPENAI_API_KEY=your_key_here                # Required
CV_GEN_OPENAI_MODEL=gpt-4                   # Optional, defaults to gpt-4
CV_GEN_LOG_LEVEL=DEBUG                      # Optional, for development
```

## Development Notes

- Always work within the activated virtual environment (.venv)
- Use Pydantic models for all data structures - never plain dictionaries
- Component-specific logging: use `get_scraping_logger()` from `utils/logging_config.py`
- LangGraph agents should inherit from base agent patterns established in existing agents
- Error handling: comprehensive try/catch with custom exception classes per module
- Type safety: full type hints required, validated by Pydantic models