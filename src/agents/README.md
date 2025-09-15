# JobAnalyzer Agent

The JobAnalyzer class provides comprehensive job posting analysis capabilities, combining web scraping with AI-powered data extraction to create structured job data.

## Overview

The JobAnalyzer agent can:

- **Analyze job postings** from both URLs and raw text input
- **Extract structured data** using OpenAI GPT models with optimized prompts
- **Validate data quality** to ensure extraction accuracy
- **Handle various error scenarios** with comprehensive retry logic
- **Provide detailed logging** for debugging and monitoring

## Core Features

### 1. Multi-Input Support
- **URL Analysis**: Automatically scrapes job posting content from URLs
- **Text Analysis**: Processes raw job posting text directly
- **Smart Detection**: Automatically determines input type (URL vs text)

### 2. AI-Powered Extraction
- **GPT Integration**: Uses OpenAI GPT-4 for intelligent data extraction
- **Structured Prompts**: Optimized prompts for consistent data extraction
- **Field Validation**: Ensures extracted data meets quality standards
- **Experience Level Mapping**: Intelligent mapping of experience requirements

### 3. Comprehensive Error Handling
- **Scraping Failures**: Graceful handling of blocked/timeout scenarios
- **API Failures**: Retry logic with exponential backoff
- **Data Validation**: Quality checks for extracted information
- **User Feedback**: Clear error messages with actionable suggestions

### 4. Data Validation
- **Quality Assessment**: Validates extracted data completeness and accuracy
- **Placeholder Detection**: Identifies low-quality extractions
- **Duplicate Checking**: Detects duplicate skills and requirements
- **Length Validation**: Ensures adequate description content

## Usage

### Basic Usage

```python
from src.agents.job_analyzer import JobAnalyzer

# Initialize analyzer
analyzer = JobAnalyzer()

# Analyze from URL
job_data = analyzer.analyze_job("https://example.com/job-posting")

# Analyze from text
job_text = "Software Engineer position at TechCorp..."
job_data = analyzer.analyze_job(job_text)

# Get analysis summary
summary = analyzer.get_analysis_summary(job_data)
```

### Advanced Configuration

```python
# Custom configuration
analyzer = JobAnalyzer(
    api_key="your-openai-key",
    model="gpt-4",
    max_retries=5,
    retry_delay=3
)

# Extract and validate
job_data = analyzer.extract_job_details(job_text)
is_valid = analyzer.validate_extraction(job_data)
```

## Data Output

The JobAnalyzer extracts the following structured data:

- **company_name**: Hiring company name
- **position**: Job title/position
- **requirements**: List of job requirements and qualifications
- **skills_required**: List of technical and soft skills
- **experience_level**: Required experience level (entry/junior/mid/senior/lead/executive/intern)
- **description**: Clean, comprehensive job description
- **url**: Source URL (if applicable)
- **extracted_at**: Extraction timestamp

## Error Handling

The JobAnalyzer handles various failure scenarios:

### Scraping Errors
- **Access Blocked**: When websites block scraping attempts
- **Timeouts**: When requests take too long
- **Invalid URLs**: When URLs are malformed or inaccessible
- **Empty Content**: When scraped content is insufficient

### API Errors
- **Authentication**: Invalid or missing OpenAI API keys
- **Rate Limiting**: When API usage limits are exceeded
- **Quota Exceeded**: When account quotas are reached
- **Network Issues**: Temporary connectivity problems

### Data Quality Issues
- **Missing Fields**: When required information cannot be extracted
- **Invalid Data**: When extracted data fails validation
- **Insufficient Content**: When job descriptions are too short
- **Placeholder Values**: When extraction returns generic placeholders

## Integration

The JobAnalyzer integrates with existing project components:

- **JobData Model**: Uses Pydantic models for structured data
- **Scraper Utils**: Leverages existing web scraping utilities
- **Logging System**: Uses component-specific logging configuration
- **Error Handling**: Follows project error handling patterns

## Testing

Comprehensive test coverage includes:

- **Unit Tests**: Individual method testing with mocking
- **Integration Tests**: End-to-end workflow testing
- **Error Scenarios**: Various failure mode testing
- **Edge Cases**: Boundary condition and invalid input testing

Run tests with:
```bash
source .venv/bin/activate
OPENAI_API_KEY=test-key python -m pytest tests/test_job_analyzer.py -v
```

## Dependencies

Required dependencies:
- `openai>=1.107.2`: OpenAI API integration
- `pydantic>=2.11.9`: Data validation and modeling
- `requests>=2.32.5`: HTTP requests for scraping
- `beautifulsoup4>=4.13.5`: HTML parsing

## Configuration

Set up the required environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or use a .env file in the project root.

## Performance Considerations

- **API Costs**: GPT-4 calls incur costs; consider using GPT-3.5-turbo for cost optimization
- **Rate Limiting**: Built-in retry logic handles rate limits gracefully
- **Caching**: Consider implementing response caching for repeated analyses
- **Batch Processing**: For multiple job postings, consider batch processing with delays

## Troubleshooting

### Common Issues

1. **"OpenAI API key not provided"**
   - Set the OPENAI_API_KEY environment variable
   - Verify the API key is valid and has sufficient quota

2. **"Job posting access blocked"**
   - The website is blocking scraping attempts
   - Try copying the job text directly instead of using the URL

3. **"Job text too short for meaningful analysis"**
   - Ensure job descriptions have at least 20 characters
   - Check if the scraped content is complete

4. **"Failed to validate extracted job data"**
   - The AI extraction may have failed
   - Check the job text for unusual formatting or content

### Debugging

Enable detailed logging by checking the logs:
- `/logs/agents.log`: Agent-specific operations
- `/logs/scraping.log`: Web scraping activities
- `/logs/errors.log`: Error messages and stack traces

## Future Enhancements

Potential improvements:
- **Multiple AI Providers**: Support for alternative LLM providers
- **Caching Layer**: Response caching for improved performance
- **Batch Processing**: Efficient handling of multiple job postings
- **Custom Prompts**: User-configurable extraction prompts
- **Data Enrichment**: Additional metadata extraction and validation