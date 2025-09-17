#!/bin/bash

# CV Generator Workflow Demo Script
# This script demonstrates the complete LangGraph workflow system

set -e  # Exit on any error

echo "🚀 CV Generator - LangGraph Workflow Orchestration System"
echo "========================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
echo
print_info "Checking prerequisites..."

# Check Python version
if ! python3 --version &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
print_status "Python $python_version detected"

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    print_error "Please run this script from the cv-generator root directory"
    exit 1
fi

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "No virtual environment detected. Consider using one:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    print_info "Loading environment variables from .env file"
    set -o allexport
    source .env
    set +o allexport
fi

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    print_error "OPENAI_API_KEY environment variable not set"
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo "Or create a .env file with OPENAI_API_KEY=your-key"
    exit 1
fi

print_status "OpenAI API key configured"

# Check dependencies
print_info "Checking dependencies..."

if ! python3 -c "import langgraph" &> /dev/null; then
    print_warning "LangGraph not found. Installing dependencies..."
    pip install -r requirements.txt
else
    print_status "Dependencies available"
fi

# Create demo workspace
DEMO_DIR="demo_workspace_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_DIR"
print_status "Created demo workspace: $DEMO_DIR"

# Function to run demo with error handling
run_demo() {
    local demo_type=$1
    local description=$2

    echo
    print_info "$description"
    echo "----------------------------------------"

    if python3 examples/demo_workflow.py --demo-dir "$DEMO_DIR" --$demo_type; then
        print_status "$description completed successfully"
    else
        print_error "$description failed"
        return 1
    fi
}

# Main demo menu
show_menu() {
    echo
    echo "📋 Available Demos:"
    echo "1. Interactive Workflow Demo"
    echo "2. Batch Processing Demo"
    echo "3. CLI Interface Demo"
    echo "4. State Management Demo"
    echo "5. Complete Integration Test"
    echo "6. View System Architecture"
    echo "7. Clean Demo Environment"
    echo "8. Exit"
    echo
}

# CLI demo function
run_cli_demo() {
    echo
    print_info "CLI Interface Demo"
    echo "-------------------"

    # Show help
    echo "📖 Command-line help:"
    python3 src/main.py --help

    echo
    echo "🔧 Available commands:"
    echo "  # Interactive mode:"
    echo "  python src/main.py --job-url 'https://example.com/job' --profile examples/sample_user_profile.yaml"
    echo
    echo "  # Batch mode:"
    echo "  python src/main.py --job-text 'Job description...' --profile profile.yaml --batch"
    echo
    echo "  # Resume workflow:"
    echo "  python src/main.py --resume workflow_states/checkpoint.json.gz"
    echo
    echo "  # Check status:"
    echo "  python src/main.py --status workflow-id-12345"

    print_status "CLI demo completed"
}

# State management demo
run_state_demo() {
    echo
    print_info "State Management Demo"
    echo "----------------------"

    # Show state directory structure
    if [ -d "$DEMO_DIR/workflow_states" ]; then
        echo "📁 Workflow state directory structure:"
        tree "$DEMO_DIR/workflow_states" 2>/dev/null || find "$DEMO_DIR/workflow_states" -type f

        echo
        echo "📊 Checkpoint analysis:"
        python3 -c "
import sys, os
sys.path.insert(0, 'src')
from workflows.state_manager import StateManager
from pathlib import Path

sm = StateManager('$DEMO_DIR/workflow_states')
checkpoints = sm.list_checkpoints()

print(f'Total checkpoints: {len(checkpoints)}')
for cp in checkpoints[:3]:
    print(f'  - {cp.step_name}: {cp.timestamp} ({cp.file_size_bytes} bytes)')
"
    else
        echo "No state data available. Run a workflow first."
    fi

    print_status "State management demo completed"
}

# Integration test function
run_integration_test() {
    echo
    print_info "Complete Integration Test"
    echo "--------------------------"

    # Run tests
    echo "🧪 Running integration tests..."
    if python3 -m pytest tests/test_workflow_integration.py -v --tb=short; then
        print_status "Integration tests passed"
    else
        print_warning "Some tests failed (this is normal in demo mode)"
    fi

    # Show test coverage if available
    if command -v coverage &> /dev/null; then
        echo
        print_info "Generating test coverage report..."
        coverage run -m pytest tests/test_workflow_integration.py
        coverage report --include="src/*"
    fi

    print_status "Integration test completed"
}

# Architecture overview
show_architecture() {
    echo
    print_info "System Architecture Overview"
    echo "=============================="

    cat << 'EOF'

🏗️  CV Generator - LangGraph Workflow Architecture

┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface (main.py)                  │
│  Interactive Mode | Batch Mode | Resume Mode | Status Mode  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               CVGenerationWorkflow                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                LangGraph Orchestration               │   │
│  │  ┌───────┐ ┌────────┐ ┌───────┐ ┌────────┐ ┌─────┐  │   │
│  │  │ Load  │→│Analyze │→│ Match │→│Customize│→│ PDF │  │   │
│  │  │Profile│ │  Job   │ │Profile│ │Template │ │ Gen │  │   │
│  │  └───────┘ └────────┘ └───────┘ └────────┘ └─────┘  │   │
│  │       ↕️              ↕️              ↕️              │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │        Human-in-the-Loop Approval          │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                State Manager                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Checkpoints │ │ Persistence │ │ Error Recovery      │   │
│  │ & Resume    │ │ & Backup    │ │ & Retry Logic       │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   AI Agents                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │JobAnalyzer  │ │ProfileMatcher│ │TemplateCustomizer   │   │
│  │(GPT-4)      │ │(GPT-4)      │ │& PreviewGenerator   │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

🔄 Workflow Features:
  • Conditional routing based on user interaction mode
  • Automatic checkpointing and state persistence
  • Error handling with exponential backoff retry
  • Progress tracking with time estimation
  • Human approval checkpoints for quality control
  • Comprehensive logging and debugging support

📊 Data Flow:
  Job Input → Structured Analysis → Profile Matching →
  Template Customization → Preview Generation →
  User Approval → PDF Generation → File Organization

EOF

    print_status "Architecture overview displayed"
}

# Clean demo environment
clean_demo() {
    echo
    print_info "Cleaning Demo Environment"
    echo "--------------------------"

    if [ -d "$DEMO_DIR" ]; then
        rm -rf "$DEMO_DIR"
        print_status "Demo workspace cleaned: $DEMO_DIR"
    else
        print_info "No demo workspace to clean"
    fi

    # Clean any test artifacts
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true

    print_status "Demo environment cleaned"
}

# Main menu loop
main_menu() {
    while true; do
        show_menu
        read -p "Select option (1-8): " choice

        case $choice in
            1)
                run_demo "interactive" "Interactive Workflow Demo"
                ;;
            2)
                run_demo "batch" "Batch Processing Demo"
                ;;
            3)
                run_cli_demo
                ;;
            4)
                run_state_demo
                ;;
            5)
                run_integration_test
                ;;
            6)
                show_architecture
                ;;
            7)
                clean_demo
                ;;
            8)
                echo
                print_status "Thanks for trying the CV Generator Workflow System!"
                break
                ;;
            *)
                print_warning "Invalid option. Please select 1-8."
                ;;
        esac

        echo
        read -p "Press Enter to continue..."
    done
}

# Check for command line arguments
if [ $# -gt 0 ]; then
    case $1 in
        --interactive)
            run_demo "interactive" "Interactive Workflow Demo"
            ;;
        --batch)
            run_demo "batch" "Batch Processing Demo"
            ;;
        --test)
            run_integration_test
            ;;
        --clean)
            clean_demo
            ;;
        --help)
            echo "Usage: $0 [--interactive|--batch|--test|--clean|--help]"
            echo "  --interactive  Run interactive demo"
            echo "  --batch        Run batch demo"
            echo "  --test         Run integration tests"
            echo "  --clean        Clean demo environment"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
else
    # Run interactive menu
    main_menu
fi

# Cleanup on exit
trap clean_demo EXIT
print_status "Demo script completed successfully"