"""
Integration tests for CV Generation Workflow.

This module provides comprehensive integration testing for the LangGraph
workflow system, including state management, agent integration, and
end-to-end workflow validation.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import workflow components
from workflows.cv_generation_workflow import CVGenerationWorkflow
from workflows.state_manager import StateManager
from workflows.workflow_state import (
    CVGenerationState, WorkflowStep, create_initial_state,
    add_log_entry, update_state_progress
)
from models.user_profile import UserProfile, PersonalInfo
from models.job_data import JobData, ExperienceLevel
from models.match_result import MatchResult, MatchScore, AnalysisDetails


class TestWorkflowState:
    """Test workflow state management and utilities."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state(
            job_input="https://example.com/job",
            user_profile_path="/path/to/profile.yaml",
            output_directory="/output",
            interactive_mode=True,
            debug_mode=False
        )

        assert state["job_input"] == "https://example.com/job"
        assert state["user_profile_path"] == "/path/to/profile.yaml"
        assert state["output_directory"] == "/output"
        assert state["interactive_mode"] is True
        assert state["debug_mode"] is False
        assert state["workflow_status"] == "initializing"
        assert state["progress_percentage"] == 0.0
        assert len(state["step_history"]) == 0
        assert len(state["errors"]) == 0

    def test_add_log_entry(self):
        """Test log entry addition."""
        state = create_initial_state("job", "profile", "output")

        add_log_entry(state, "INFO", "Test message", "test_step", {"key": "value"})

        assert len(state["log_entries"]) == 1
        entry = state["log_entries"][0]
        assert entry["level"] == "INFO"
        assert entry["message"] == "Test message"
        assert entry["step"] == "test_step"
        assert entry["details"]["key"] == "value"

    def test_workflow_step_lifecycle(self):
        """Test WorkflowStep lifecycle methods."""
        step = WorkflowStep(name="test_step")

        assert step.status == "pending"
        assert step.start_time is None

        # Start step
        step.start()
        assert step.status == "running"
        assert step.start_time is not None

        # Complete step
        step.complete()
        assert step.status == "completed"
        assert step.end_time is not None
        assert step.duration_seconds is not None

    def test_workflow_step_failure(self):
        """Test WorkflowStep failure handling."""
        step = WorkflowStep(name="test_step")
        step.start()

        step.fail("Test error message")
        assert step.status == "failed"
        assert step.error_message == "Test error message"
        assert step.end_time is not None


class TestStateManager:
    """Test state persistence and management."""

    def test_state_manager_initialization(self):
        """Test StateManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_manager = StateManager(
                state_directory=temp_dir,
                max_checkpoints=5,
                compress_states=True
            )

            assert state_manager.state_directory == Path(temp_dir)
            assert state_manager.max_checkpoints == 5
            assert state_manager.compress_states is True

            # Check directories were created
            assert state_manager.checkpoints_dir.exists()
            assert state_manager.backups_dir.exists()
            assert state_manager.temp_dir.exists()

    def test_save_and_load_state(self):
        """Test state saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_manager = StateManager(state_directory=temp_dir)

            # Create test state
            state = create_initial_state("job", "profile", "output")
            state["current_step"] = "test_step"
            state["progress_percentage"] = 50.0

            # Save state
            saved_path = state_manager.save_state(state, "test_checkpoint")
            assert Path(saved_path).exists()

            # Load state
            loaded_state = state_manager.load_state(saved_path)
            assert loaded_state["workflow_id"] == state["workflow_id"]
            assert loaded_state["current_step"] == "test_step"
            assert loaded_state["progress_percentage"] == 50.0

    def test_list_checkpoints(self):
        """Test checkpoint listing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_manager = StateManager(state_directory=temp_dir)

            # Create and save multiple states
            state1 = create_initial_state("job1", "profile", "output")
            state2 = create_initial_state("job2", "profile", "output")

            state_manager.save_state(state1, "checkpoint1")
            state_manager.save_state(state2, "checkpoint2")

            # List all checkpoints
            all_checkpoints = state_manager.list_checkpoints()
            assert len(all_checkpoints) == 2

            # List checkpoints for specific workflow
            workflow1_checkpoints = state_manager.list_checkpoints(state1["workflow_id"])
            assert len(workflow1_checkpoints) == 1
            assert workflow1_checkpoints[0].workflow_id == state1["workflow_id"]


class TestWorkflowIntegration:
    """Test complete workflow integration with mocked agents."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(state_directory=self.temp_dir)

        # Mock configuration
        self.config = {
            'openai_model': 'gpt-4',
            'max_retries': 2,
            'templates_dir': Path(self.temp_dir) / "templates",
            'output_dir': Path(self.temp_dir) / "output"
        }

        # Create template directory
        self.config['templates_dir'].mkdir(parents=True, exist_ok=True)
        self.config['output_dir'].mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_user_profile(self):
        """Create mock user profile."""
        return UserProfile(
            personal_info=PersonalInfo(
                name="John Doe",
                email="john@example.com",
                phone="+1234567890",
                location="San Francisco, CA"
            ),
            experiences=[],
            skills=["Python", "JavaScript", "React"],
            education=[],
            projects=[]
        )

    @pytest.fixture
    def mock_job_data(self):
        """Create mock job data."""
        return JobData(
            company_name="Tech Corp",
            position="Software Engineer",
            requirements=["Bachelor's degree", "3+ years experience"],
            skills_required=["Python", "JavaScript", "SQL"],
            experience_level=ExperienceLevel.MID,
            description="Great opportunity for a software engineer"
        )

    @pytest.fixture
    def mock_match_result(self):
        """Create mock match result."""
        return MatchResult(
            score=MatchScore.GOOD,
            matched_skills=["Python", "JavaScript"],
            relevant_experiences=[],
            recommended_projects=[],
            suggestions=[],
            analysis_details=AnalysisDetails(
                skills_match_percentage=75.0,
                experience_match_percentage=60.0
            ),
            job_title="Software Engineer",
            company_name="Tech Corp"
        )

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    async def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = CVGenerationWorkflow(
            config=self.config,
            state_manager=self.state_manager,
            debug_mode=True
        )

        assert workflow.config == self.config
        assert workflow.debug_mode is True
        assert workflow.state_manager == self.state_manager
        assert hasattr(workflow, 'job_analyzer')
        assert hasattr(workflow, 'profile_matcher')
        assert hasattr(workflow, 'template_customizer')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('workflows.cv_generation_workflow.load_user_profile_from_yaml')
    async def test_load_user_profile_node(self, mock_load_profile, mock_user_profile):
        """Test user profile loading node."""
        mock_load_profile.return_value = mock_user_profile

        workflow = CVGenerationWorkflow(config=self.config, debug_mode=True)
        state = create_initial_state("job", str(Path(self.temp_dir) / "profile.yaml"), "output")

        # Create dummy profile file
        profile_path = Path(state["user_profile_path"])
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.touch()

        result_state = await workflow.load_user_profile_node(state)

        assert result_state["user_profile"] == mock_user_profile
        assert result_state["current_step"] == "analyze_job"
        assert len(result_state["step_history"]) == 1
        assert result_state["step_history"][0].status == "completed"

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    async def test_analyze_job_node_with_mock(self, mock_job_data):
        """Test job analysis node with mocked analyzer."""
        workflow = CVGenerationWorkflow(config=self.config, debug_mode=True)

        # Mock the job analyzer
        workflow.job_analyzer.analyze_job = AsyncMock(return_value=mock_job_data)

        state = create_initial_state("https://example.com/job", "profile", "output")
        result_state = await workflow.analyze_job_node(state)

        assert result_state["job_data"] == mock_job_data
        assert result_state["current_step"] == "match_profile"
        workflow.job_analyzer.analyze_job.assert_called_once_with("https://example.com/job")

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    async def test_match_profile_node_with_mock(self, mock_user_profile, mock_job_data, mock_match_result):
        """Test profile matching node with mocked matcher."""
        workflow = CVGenerationWorkflow(config=self.config, debug_mode=True)

        # Mock the profile matcher
        workflow.profile_matcher.analyze_compatibility = AsyncMock(return_value=mock_match_result)

        state = create_initial_state("job", "profile", "output")
        state["user_profile"] = mock_user_profile
        state["job_data"] = mock_job_data

        result_state = await workflow.match_profile_node(state)

        assert result_state["match_result"] == mock_match_result
        assert result_state["current_step"] == "customize_templates"
        workflow.profile_matcher.analyze_compatibility.assert_called_once_with(
            mock_user_profile, mock_job_data
        )

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    async def test_error_handling_and_retry(self):
        """Test error handling and retry logic."""
        workflow = CVGenerationWorkflow(config=self.config, debug_mode=True)

        # Mock job analyzer to fail first time, succeed second time
        call_count = 0

        def mock_analyze_job(job_input):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return JobData(
                company_name="Test Corp",
                position="Test Position",
                description="Test description"
            )

        workflow.job_analyzer.analyze_job = Mock(side_effect=mock_analyze_job)

        state = create_initial_state("job_text", "profile", "output")
        result_state = await workflow.analyze_job_node(state)

        # Should succeed after retry
        assert result_state["job_data"] is not None
        assert call_count == 2  # One failure, one success
        assert len(result_state["errors"]) == 1  # One error logged
        assert result_state["retry_attempts"]["analyze_job"] == 1

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    async def test_human_approval_workflow(self, mock_user_profile, mock_job_data, mock_match_result):
        """Test human approval workflow."""
        workflow = CVGenerationWorkflow(config=self.config, debug_mode=True)

        # Mock preview generator
        mock_preview_result = Mock()
        mock_preview_result.cv_preview = "CV Preview"
        mock_preview_result.cover_letter_preview = "Cover Letter Preview"
        mock_preview_result.match_summary = "Match Summary"
        mock_preview_result.preview_html = "<html>Preview</html>"

        workflow.preview_generator.generate_preview = Mock(return_value=mock_preview_result)

        state = create_initial_state("job", "profile", "output")
        state["customized_templates"] = {
            "cv_html": "<html>CV</html>",
            "cover_letter_html": "<html>Cover Letter</html>"
        }
        state["job_data"] = mock_job_data
        state["match_result"] = mock_match_result
        state["interactive_mode"] = True

        result_state = await workflow.generate_preview_node(state)

        assert result_state["preview_content"] is not None
        assert "cv_preview" in result_state["preview_content"]
        assert "cover_letter_preview" in result_state["preview_content"]

        # Test approval request creation
        approval_state = await workflow.human_approval_node(result_state)
        assert approval_state["approval_request"] is not None
        assert approval_state["workflow_status"] == "paused"

    def test_conditional_routing_functions(self):
        """Test conditional routing logic."""
        workflow = CVGenerationWorkflow(config=self.config, debug_mode=True)

        # Test approval routing
        interactive_state = create_initial_state("job", "profile", "output", interactive_mode=True)
        assert workflow.should_request_approval(interactive_state) == "request_approval"

        batch_state = create_initial_state("job", "profile", "output", interactive_mode=False)
        assert workflow.should_request_approval(batch_state) == "skip_approval"

        # Test approval response routing
        approved_state = create_initial_state("job", "profile", "output")
        approved_state["approval_response"] = {"action": "approve"}
        assert workflow.check_approval_response(approved_state) == "approved"

        rejected_state = create_initial_state("job", "profile", "output")
        rejected_state["approval_response"] = {"action": "reject"}
        assert workflow.check_approval_response(rejected_state) == "rejected"


class TestEndToEndWorkflow:
    """End-to-end workflow testing with comprehensive mocking."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.profile_path = Path(self.temp_dir) / "test_profile.yaml"
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal profile file
        self.profile_path.write_text("""
personal_info:
  name: "John Doe"
  email: "john@example.com"
skills:
  - "Python"
  - "JavaScript"
""")

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('workflows.cv_generation_workflow.CVGenerationWorkflow._get_user_approval')
    async def test_complete_interactive_workflow(self, mock_get_approval):
        """Test complete interactive workflow execution."""
        # Mock user approval
        mock_get_approval.return_value = {"action": "approve"}

        config = {
            'openai_model': 'gpt-4',
            'templates_dir': Path(self.temp_dir) / "templates",
            'output_dir': self.output_dir
        }

        # Create templates directory
        config['templates_dir'].mkdir(exist_ok=True)

        workflow = CVGenerationWorkflow(config=config, debug_mode=True)

        # Mock all agents
        workflow.job_analyzer.analyze_job = AsyncMock(return_value=JobData(
            company_name="Test Corp",
            position="Software Engineer",
            description="Test job description"
        ))

        workflow.profile_matcher.analyze_compatibility = AsyncMock(return_value=MatchResult(
            score=MatchScore.GOOD,
            matched_skills=["Python"],
            analysis_details=AnalysisDetails(
                skills_match_percentage=75.0,
                experience_match_percentage=60.0
            ),
            job_title="Software Engineer",
            company_name="Test Corp"
        ))

        mock_customization_result = Mock()
        mock_customization_result.cv_html = "<html>Custom CV</html>"
        mock_customization_result.cover_letter_html = "<html>Custom Cover Letter</html>"
        mock_customization_result.changes_made = ["Change 1", "Change 2"]
        mock_customization_result.customization_score = 0.8

        workflow.template_customizer.customize_for_job = Mock(return_value=mock_customization_result)

        mock_preview_result = Mock()
        mock_preview_result.cv_preview = "CV Preview"
        mock_preview_result.cover_letter_preview = "Cover Letter Preview"
        mock_preview_result.match_summary = "Good match"
        mock_preview_result.preview_html = "<html>Preview</html>"

        workflow.preview_generator.generate_preview = Mock(return_value=mock_preview_result)

        mock_pdf_result = Mock()
        mock_pdf_result["cv"] = str(self.output_dir / "cv.pdf")
        mock_pdf_result["cover_letter"] = str(self.output_dir / "cover_letter.pdf")
        mock_pdf_result.preview_html_path = None
        mock_pdf_result.file_sizes = {"cv": 1024, "cover_letter": 512}
        mock_pdf_result.generation_time_seconds = 2.5

        workflow.pdf_generator.generate_pdfs = AsyncMock(return_value=mock_pdf_result)

        # Run workflow
        result = await workflow.run_workflow(
            job_input="Test job description",
            user_profile_path=str(self.profile_path),
            output_directory=str(self.output_dir),
            interactive_mode=True
        )

        # Verify results
        assert result["status"] == "completed"
        assert result["progress"] == 100.0
        assert "output_files" in result
        assert result["match_score"] == MatchScore.GOOD
        assert len(result["errors"]) == 0

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    async def test_batch_mode_workflow(self):
        """Test batch mode workflow execution."""
        config = {
            'openai_model': 'gpt-4',
            'templates_dir': Path(self.temp_dir) / "templates",
            'output_dir': self.output_dir
        }

        config['templates_dir'].mkdir(exist_ok=True)

        workflow = CVGenerationWorkflow(config=config, debug_mode=True)

        # Mock agents for batch mode (no human interaction)
        workflow.job_analyzer.analyze_job = AsyncMock(return_value=JobData(
            company_name="Batch Corp",
            position="Data Scientist",
            description="Batch job description"
        ))

        workflow.profile_matcher.analyze_compatibility = AsyncMock(return_value=MatchResult(
            score=MatchScore.EXCELLENT,
            matched_skills=["Python", "JavaScript"],
            analysis_details=AnalysisDetails(
                skills_match_percentage=90.0,
                experience_match_percentage=85.0
            ),
            job_title="Data Scientist",
            company_name="Batch Corp"
        ))

        # Mock other agents...
        mock_customization_result = Mock()
        mock_customization_result.cv_html = "<html>Batch CV</html>"
        mock_customization_result.cover_letter_html = "<html>Batch Cover Letter</html>"
        mock_customization_result.changes_made = ["Batch Change"]
        mock_customization_result.customization_score = 0.9

        workflow.template_customizer.customize_for_job = Mock(return_value=mock_customization_result)

        # In batch mode, preview should be skipped
        mock_pdf_result = Mock()
        mock_pdf_result["cv"] = str(self.output_dir / "batch_cv.pdf")
        mock_pdf_result["cover_letter"] = str(self.output_dir / "batch_cover_letter.pdf")
        mock_pdf_result.preview_html_path = None
        mock_pdf_result.file_sizes = {"cv": 2048, "cover_letter": 1024}
        mock_pdf_result.generation_time_seconds = 1.8

        workflow.pdf_generator.generate_pdfs = AsyncMock(return_value=mock_pdf_result)

        # Run batch workflow
        result = await workflow.run_workflow(
            job_input="Batch job description",
            user_profile_path=str(self.profile_path),
            output_directory=str(self.output_dir),
            interactive_mode=False
        )

        # Verify batch results
        assert result["status"] == "completed"
        assert result["match_score"] == MatchScore.EXCELLENT


# Pytest configuration and runners
@pytest.mark.asyncio
class TestAsyncWorkflow:
    """Test async workflow functionality."""

    async def test_workflow_concurrency(self):
        """Test workflow handles concurrent operations correctly."""
        # This would test if multiple workflows can run concurrently
        # without interfering with each other
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])