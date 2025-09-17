# CV Generator - LangGraph Workflow Orchestration System

A comprehensive CV/Cover Letter generation system built with LangGraph workflow orchestration, featuring human-in-the-loop processes, state management, and intelligent agent coordination.

## 🌟 Features

### Core Workflow Capabilities
- **LangGraph Orchestration**: Complete workflow management with conditional routing
- **Human-in-the-Loop**: Interactive approval checkpoints for quality control
- **State Persistence**: Resume workflows from any checkpoint
- **Error Recovery**: Automatic retry logic with exponential backoff
- **Progress Tracking**: Real-time progress monitoring with time estimation

### Agent Integration
- **Job Analyzer**: AI-powered job posting analysis and structured data extraction
- **Profile Matcher**: Intelligent compatibility scoring between profiles and jobs
- **Template Customizer**: Dynamic CV/Cover Letter customization based on job requirements
- **Preview Generator**: Interactive preview generation with match analysis
- **PDF Generator**: Professional PDF document creation with optimized formatting

### Interface Modes
- **Interactive Mode**: Step-by-step workflow with user approval
- **Batch Mode**: Automated processing without human intervention
- **Resume Mode**: Continue from saved checkpoints
- **Status Monitoring**: Real-time workflow status checking

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**:
   ```bash
   python >= 3.8
   ```

2. **Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

### Basic Usage

#### Interactive Mode
```bash
# Start interactive workflow with job URL
python src/main.py --job-url "https://example.com/job-posting" \
                   --profile examples/sample_user_profile.yaml \
                   --interactive

# Use job text from file
python src/main.py --job-text "@path/to/job_description.txt" \
                   --profile examples/sample_user_profile.yaml \
                   --interactive
```

#### Batch Mode
```bash
# Automated processing without human interaction
python src/main.py --job-text "Software Engineer position..." \
                   --profile examples/sample_user_profile.yaml \
                   --batch \
                   --output ./applications
```

#### Resume Workflow
```bash
# Resume from checkpoint
python src/main.py --resume workflow_states/checkpoint_xyz.json.gz

# Check workflow status
python src/main.py --status workflow-id-12345
```

### Demo Mode
```bash
# Run interactive demo
python examples/demo_workflow.py --interactive

# Run batch demo
python examples/demo_workflow.py --batch
```

## 📋 Workflow Steps

The system executes the following workflow steps:

1. **Load User Profile** (`load_user_profile`)
   - Parse and validate YAML profile
   - Validate required fields and data integrity

2. **Analyze Job** (`analyze_job`)
   - Extract structured data from job posting
   - Identify required skills, experience level, and qualifications

3. **Match Profile** (`match_profile`)
   - Calculate compatibility score (1-10)
   - Identify relevant experiences and projects
   - Generate improvement suggestions

4. **Customize Templates** (`customize_templates`)
   - Personalize CV and cover letter templates
   - Prioritize content based on job requirements
   - Optimize for single-page PDF constraints

5. **Generate Preview** (`generate_preview`)
   - Create interactive preview with match analysis
   - Generate content summaries for review

6. **Human Approval** (`human_approval`) - Interactive Mode Only
   - Present preview to user for approval
   - Allow modifications or rejection
   - Support iterative refinement

7. **Generate PDFs** (`generate_pdfs`)
   - Create professional PDF documents
   - Apply consistent formatting and styling
   - Organize output files

8. **Cleanup** (`cleanup`)
   - Organize generated files
   - Clean up temporary resources
   - Create workflow summary

## 🔧 Configuration

### User Profile Format

Create a YAML file with your professional information:

```yaml
personal_info:
  name: "Your Name"
  email: "your.email@example.com"
  phone: "+1 (555) 123-4567"
  location: "City, State"

skills:
  - "Python"
  - "JavaScript"
  - "React"
  # ... more skills

experiences:
  - company: "Company Name"
    role: "Job Title"
    start_date: "2020-01-01"
    end_date: null  # null for current position
    achievements:
      - "Achievement 1"
      - "Achievement 2"
    technologies:
      - "Technology 1"
      - "Technology 2"

education:
  - degree: "Bachelor of Science"
    field_of_study: "Computer Science"
    institution: "University Name"
    graduation_year: 2020

projects:
  - title: "Project Name"
    description: "Project description"
    technologies: ["Tech1", "Tech2"]
    url: "https://github.com/user/project"
    status: "completed"
```

### Workflow Configuration

```python
config = {
    'openai_model': 'gpt-4',          # OpenAI model to use
    'max_retries': 3,                 # Maximum retry attempts
    'templates_dir': './templates',    # Template directory
    'output_dir': './applications',    # Output directory
    'state_dir': './workflow_states', # State persistence directory
}
```

## 🧪 Testing

### Run Integration Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_workflow_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Workflow Components
```bash
# Test individual agents
python -m pytest tests/agents/ -v

# Test state management
python -m pytest tests/test_workflow_integration.py::TestStateManager -v

# Test end-to-end workflow
python -m pytest tests/test_workflow_integration.py::TestEndToEndWorkflow -v
```

## 📁 Project Structure

```
cv-generator/
├── src/
│   ├── main.py                 # Main CLI application
│   ├── workflows/              # LangGraph workflow system
│   │   ├── cv_generation_workflow.py    # Main workflow orchestration
│   │   ├── state_manager.py             # State persistence & recovery
│   │   └── workflow_state.py            # State schema & utilities
│   ├── agents/                 # AI agents
│   │   ├── job_analyzer.py     # Job posting analysis
│   │   ├── profile_matcher.py  # Profile-job matching
│   │   ├── template_customizer.py      # Template personalization
│   │   ├── preview_generator.py        # Preview generation
│   │   └── pdf_generator.py            # PDF document creation
│   ├── models/                 # Data models
│   │   ├── user_profile.py     # User profile schema
│   │   ├── job_data.py         # Job data schema
│   │   └── match_result.py     # Match analysis results
│   └── utils/                  # Utilities
│       ├── logging_config.py   # Logging setup
│       ├── file_manager.py     # File operations
│       └── yaml_parser.py      # YAML processing
├── tests/                      # Test suite
│   ├── test_workflow_integration.py    # Integration tests
│   └── agents/                 # Agent-specific tests
├── examples/                   # Examples & demos
│   ├── sample_user_profile.yaml        # Sample profile
│   └── demo_workflow.py                # Interactive demo
├── templates/                  # CV/Cover letter templates
├── applications/               # Generated applications
├── workflow_states/            # Workflow checkpoints
└── requirements.txt            # Dependencies
```

## 🔄 State Management

### Checkpoint System
- Automatic checkpoints at each workflow step
- Compressed JSON storage with metadata
- Configurable retention policies
- State validation and migration support

### Resume Functionality
```bash
# List available checkpoints
python src/main.py --status workflow-id

# Resume from specific checkpoint
python src/main.py --resume path/to/checkpoint.json.gz

# Resume latest checkpoint for workflow
python src/main.py --resume-latest workflow-id
```

### State Directory Structure
```
workflow_states/
├── checkpoints/
│   ├── workflow-123_step1_20240916_143022.json.gz
│   └── workflow-123_step2_20240916_143145.json.gz
├── backups/
│   └── backup_20240916_143000_checkpoint.json.gz
└── temp/
    └── temp_state_processing.json
```

## 🛠️ Advanced Features

### Error Recovery
- Automatic retry with exponential backoff
- Step-level error isolation
- Graceful degradation strategies
- Detailed error logging and context

### Progress Tracking
- Real-time progress percentage calculation
- Time estimation based on historical data
- Step-by-step status monitoring
- Performance metrics collection

### Human-in-the-Loop
- Interactive approval checkpoints
- Content preview with match analysis
- Modification and refinement options
- Approval history tracking

### Batch Processing
- Multiple job processing pipeline
- Resource optimization for large batches
- Parallel workflow execution
- Bulk output organization

## 🎯 Use Cases

### Individual Job Applications
```bash
# Single job application with review
python src/main.py --job-url "https://company.com/careers/123" \
                   --profile my_profile.yaml \
                   --interactive
```

### Bulk Application Generation
```bash
# Process multiple jobs in batch
for job_file in jobs/*.txt; do
  python src/main.py --job-text "@$job_file" \
                     --profile my_profile.yaml \
                     --batch \
                     --output "applications/$(basename $job_file .txt)"
done
```

### Workflow Automation
```python
# Python API usage
from workflows.cv_generation_workflow import CVGenerationWorkflow
from workflows.state_manager import StateManager

async def automated_application():
    workflow = CVGenerationWorkflow(config=my_config)

    result = await workflow.run_workflow(
        job_input=job_text,
        user_profile_path="profile.yaml",
        output_directory="./output",
        interactive_mode=False
    )

    return result
```

## 📊 Monitoring and Analytics

### Workflow Metrics
- Processing time per step
- Success/failure rates
- Retry statistics
- Resource utilization

### Quality Metrics
- Match score distributions
- User approval rates
- Template customization effectiveness
- PDF generation success rates

### Performance Optimization
- Checkpoint frequency tuning
- Retry strategy optimization
- Resource cleanup scheduling
- Cache management

## 🚨 Troubleshooting

### Common Issues

**OpenAI API Errors**:
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test API connectivity
python -c "from openai import OpenAI; print(OpenAI().models.list().data[0].id)"
```

**Profile Validation Errors**:
```bash
# Validate profile format
python -c "from utils.yaml_parser import load_user_profile_from_yaml; load_user_profile_from_yaml('profile.yaml')"
```

**Workflow State Issues**:
```bash
# Check state directory permissions
ls -la workflow_states/

# Validate checkpoint integrity
python -c "from workflows.state_manager import StateManager; sm = StateManager('./workflow_states'); print(len(sm.list_checkpoints()))"
```

### Debug Mode
```bash
# Enable detailed logging
python src/main.py --debug --log-level DEBUG ...

# Preserve temporary files
python src/main.py --no-cleanup ...
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd cv-generator

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Code formatting
black src/ tests/
flake8 src/ tests/
```

### Adding New Features
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review test examples
- Use debug mode for troubleshooting

---

**Built with ❤️ using LangGraph, OpenAI, and Python**