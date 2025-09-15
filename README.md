# CV Generator

A sophisticated Python application for generating customized CVs and cover letters using AI agents and LangGraph workflows.

## Features

- **AI-Powered Generation**: Uses OpenAI GPT models to create personalized CVs and cover letters
- **Template System**: Supports customizable Jinja2 templates for different CV formats
- **Web Scraping**: Extracts job requirements from job postings using BeautifulSoup and Playwright
- **Agent Architecture**: Built with LangGraph for complex workflow management
- **Professional Logging**: Comprehensive logging with rotation and multiple output formats
- **Type Safety**: Full Pydantic models for data validation and type safety

## Project Structure

```
cv-generator/
├── src/
│   ├── agents/          # LangGraph agents for CV generation workflows
│   ├── models/          # Pydantic data models
│   ├── utils/           # Utility functions and logging configuration
│   └── main.py          # Main application entry point
├── config/              # Configuration files and settings
├── templates/           # Jinja2 templates for CV and cover letter formats
├── applications/        # Generated CVs and cover letters (gitignored)
├── logs/                # Application logs (gitignored)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd cv-generator
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Install uv if you don't have it
   pip install uv

   # Install project dependencies
   uv pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp config/.env.example config/.env

   # Edit config/.env and add your OpenAI API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Install Playwright browsers** (for web scraping):
   ```bash
   playwright install
   ```

## Usage

### Basic Usage

Run the application:
```bash
cd src
python main.py
```

### Command Line Options

```bash
python main.py --help                 # Show help message
python main.py --version              # Show version information
python main.py --debug                # Run in debug mode
python main.py --log-level DEBUG      # Set specific log level
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) | - |
| `CV_GEN_OPENAI_MODEL` | OpenAI model to use | `gpt-4` |
| `CV_GEN_OPENAI_TEMPERATURE` | Temperature for text generation | `0.7` |
| `CV_GEN_OPENAI_MAX_TOKENS` | Maximum tokens for generation | `2000` |
| `CV_GEN_LOG_LEVEL` | Logging level | `INFO` |
| `CV_GEN_LOG_FILE` | Log file name | `cv_generator.log` |
| `CV_GEN_DEBUG` | Enable debug mode | `false` |

## Development

### Project Architecture

The application follows a clean architecture pattern:

- **Configuration**: Centralized configuration management with Pydantic models
- **Logging**: Professional logging setup with file rotation and console output
- **Type Safety**: Full type hints and Pydantic models for data validation
- **Modularity**: Clear separation of concerns with dedicated modules

### Key Dependencies

- **LangGraph**: Workflow orchestration and agent management
- **OpenAI**: GPT model integration for text generation
- **Pydantic**: Data validation and settings management
- **Jinja2**: Template rendering for CV formats
- **BeautifulSoup4**: HTML parsing for web scraping
- **Playwright**: Browser automation for dynamic content
- **PyYAML**: Configuration file parsing

### Adding New Features

1. **Data Models**: Add new Pydantic models in `src/models/`
2. **Agents**: Implement LangGraph agents in `src/agents/`
3. **Templates**: Create Jinja2 templates in `templates/`
4. **Configuration**: Extend configuration in `config/config.py`

## Configuration

The application uses a hierarchical configuration system:

1. **Environment variables** (highest priority)
2. **Configuration files** in `config/`
3. **Default values** in the code

### Configuration Files

- `config/config.py`: Main configuration module
- `config/.env.example`: Example environment file
- `config/.env`: Your local environment file (create from example)

## Logging

Logs are written to both console and file with automatic rotation:

- **Console**: Colored output for development
- **File**: Structured logs in `logs/cv_generator.log`
- **Rotation**: Automatic rotation at 10MB with 5 backup files

## Security

- API keys are stored in environment variables
- Sensitive files are excluded from version control
- Input validation using Pydantic models

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Add tests for new functionality
5. Update documentation as needed

## License

[Add your license information here]

## Support

[Add support information here]