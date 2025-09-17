"""
Workflow state schema for CV generation pipeline.

This module defines the comprehensive state schema for the LangGraph workflow,
including all data models, control flags, and error handling structures.
"""

from typing import TypedDict, List, Dict, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.user_profile import UserProfile
from models.job_data import JobData
from models.match_result import MatchResult


class WorkflowStep(BaseModel):
    """Model for tracking individual workflow steps."""

    name: str = Field(..., description="Step name")
    status: Literal["pending", "running", "completed", "failed", "skipped"] = Field(
        default="pending", description="Step status"
    )
    start_time: Optional[datetime] = Field(default=None, description="Step start time")
    end_time: Optional[datetime] = Field(default=None, description="Step completion time")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    duration_seconds: Optional[float] = Field(default=None, description="Step duration")

    def start(self) -> None:
        """Mark step as started."""
        self.status = "running"
        self.start_time = datetime.now()

    def complete(self) -> None:
        """Mark step as completed."""
        self.status = "completed"
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def fail(self, error_message: str) -> None:
        """Mark step as failed."""
        self.status = "failed"
        self.end_time = datetime.now()
        self.error_message = error_message
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def retry(self) -> None:
        """Increment retry count and reset to running."""
        self.retry_count += 1
        self.status = "running"
        self.start_time = datetime.now()
        self.error_message = None


class ValidationResult(BaseModel):
    """Model for validation results."""

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    score: Optional[float] = Field(default=None, description="Validation score (0-1)")


class HumanApprovalRequest(BaseModel):
    """Model for human approval requests."""

    step_name: str = Field(..., description="Step requesting approval")
    content_preview: str = Field(..., description="Content to be approved")
    approval_type: Literal["preview", "customization", "final"] = Field(
        ..., description="Type of approval needed"
    )
    options: List[str] = Field(
        default=["approve", "reject", "modify"], description="Available options"
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    requested_at: datetime = Field(default_factory=datetime.now, description="Request timestamp")


class CVGenerationState(TypedDict):
    """
    Comprehensive state schema for CV generation workflow.

    This TypedDict defines all the data and control structures needed
    for the complete CV generation pipeline, including inputs, outputs,
    intermediate results, and workflow control.
    """

    # Input Configuration
    job_input: str  # URL or raw text
    user_profile_path: str  # Path to user profile YAML
    output_directory: str  # Output directory path
    config: Dict[str, Any]  # Workflow configuration

    # Loaded Data
    user_profile: Optional[UserProfile]  # Loaded user profile
    job_data: Optional[JobData]  # Analyzed job data
    match_result: Optional[MatchResult]  # Profile-job match analysis

    # Template Customization Results
    customized_templates: Optional[Dict[str, str]]  # Customized HTML templates
    preview_content: Optional[Dict[str, str]]  # Preview content for approval
    final_pdfs: Optional[Dict[str, str]]  # Generated PDF file paths

    # Workflow Control
    current_step: str  # Current workflow step
    workflow_status: Literal["initializing", "running", "paused", "completed", "failed"]
    step_history: List[WorkflowStep]  # History of completed steps

    # Human Interaction
    approval_request: Optional[HumanApprovalRequest]  # Pending approval request
    approval_response: Optional[Dict[str, Any]]  # User's approval response
    interactive_mode: bool  # Whether running in interactive mode

    # Error Handling
    errors: List[str]  # List of errors encountered
    warnings: List[str]  # List of warnings
    retry_attempts: Dict[str, int]  # Retry counts by step
    max_retries: int  # Maximum retry attempts per step

    # Validation Results
    profile_validation: Optional[ValidationResult]  # Profile validation result
    job_validation: Optional[ValidationResult]  # Job data validation result
    template_validation: Optional[ValidationResult]  # Template validation result

    # Progress Tracking
    start_time: datetime  # Workflow start time
    last_update: datetime  # Last state update time
    progress_percentage: float  # Overall progress (0-100)
    estimated_completion: Optional[datetime]  # Estimated completion time

    # File Management
    temp_files: List[str]  # Temporary files to clean up
    output_files: List[str]  # Generated output files
    backup_state_path: Optional[str]  # Path to backup state file

    # Debugging and Logging
    debug_mode: bool  # Whether debug mode is enabled
    log_entries: List[Dict[str, Any]]  # Detailed log entries
    performance_metrics: Dict[str, float]  # Performance metrics by step

    # Resume/Persistence
    workflow_id: str  # Unique workflow identifier
    checkpoint_data: Optional[Dict[str, Any]]  # Checkpoint data for resuming
    state_version: str  # State schema version for compatibility


def create_initial_state(
    job_input: str,
    user_profile_path: str,
    output_directory: str,
    config: Optional[Dict[str, Any]] = None,
    interactive_mode: bool = True,
    debug_mode: bool = False
) -> CVGenerationState:
    """
    Create an initial workflow state with default values.

    Args:
        job_input: Job URL or raw text
        user_profile_path: Path to user profile YAML
        output_directory: Output directory path
        config: Optional workflow configuration
        interactive_mode: Whether to run in interactive mode
        debug_mode: Whether to enable debug mode

    Returns:
        Initial CVGenerationState
    """
    import uuid

    now = datetime.now()
    workflow_id = str(uuid.uuid4())

    return CVGenerationState(
        # Input Configuration
        job_input=job_input,
        user_profile_path=user_profile_path,
        output_directory=output_directory,
        config=config or {},

        # Loaded Data
        user_profile=None,
        job_data=None,
        match_result=None,

        # Template Customization Results
        customized_templates=None,
        preview_content=None,
        final_pdfs=None,

        # Workflow Control
        current_step="initializing",
        workflow_status="initializing",
        step_history=[],

        # Human Interaction
        approval_request=None,
        approval_response=None,
        interactive_mode=interactive_mode,

        # Error Handling
        errors=[],
        warnings=[],
        retry_attempts={},
        max_retries=3,

        # Validation Results
        profile_validation=None,
        job_validation=None,
        template_validation=None,

        # Progress Tracking
        start_time=now,
        last_update=now,
        progress_percentage=0.0,
        estimated_completion=None,

        # File Management
        temp_files=[],
        output_files=[],
        backup_state_path=None,

        # Debugging and Logging
        debug_mode=debug_mode,
        log_entries=[],
        performance_metrics={},

        # Resume/Persistence
        workflow_id=workflow_id,
        checkpoint_data=None,
        state_version="1.0"
    )


def get_step_progress_weights() -> Dict[str, float]:
    """
    Get the progress weights for each workflow step.

    Returns:
        Dictionary mapping step names to their progress weights (0-1)
    """
    return {
        "load_user_profile": 0.05,
        "analyze_job": 0.15,
        "match_profile": 0.20,
        "customize_templates": 0.25,
        "generate_preview": 0.10,
        "human_approval": 0.05,
        "generate_pdfs": 0.15,
        "cleanup": 0.05
    }


def calculate_progress_percentage(step_history: List[WorkflowStep]) -> float:
    """
    Calculate overall progress percentage based on completed steps.

    Args:
        step_history: List of workflow steps with their status

    Returns:
        Progress percentage (0-100)
    """
    weights = get_step_progress_weights()
    total_progress = 0.0

    for step in step_history:
        if step.name in weights:
            if step.status == "completed":
                total_progress += weights[step.name]
            elif step.status == "running":
                total_progress += weights[step.name] * 0.5  # 50% for running steps

    return min(total_progress * 100, 100.0)


def update_state_progress(state: CVGenerationState) -> None:
    """
    Update the progress tracking in the state.

    Args:
        state: The workflow state to update
    """
    state["progress_percentage"] = calculate_progress_percentage(state["step_history"])
    state["last_update"] = datetime.now()

    # Estimate completion time based on current progress and elapsed time
    if state["progress_percentage"] > 0:
        elapsed = (datetime.now() - state["start_time"]).total_seconds()
        total_estimated = elapsed / (state["progress_percentage"] / 100)
        remaining_seconds = total_estimated - elapsed

        if remaining_seconds > 0:
            from datetime import timedelta
            state["estimated_completion"] = datetime.now() + timedelta(seconds=remaining_seconds)


def add_log_entry(
    state: CVGenerationState,
    level: str,
    message: str,
    step: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a log entry to the state.

    Args:
        state: The workflow state
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        message: Log message
        step: Optional step name
        details: Optional additional details
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "step": step,
        "details": details or {}
    }
    state["log_entries"].append(entry)

    # Keep only last 1000 log entries to prevent memory issues
    if len(state["log_entries"]) > 1000:
        state["log_entries"] = state["log_entries"][-1000:]


def get_current_step_info(state: CVGenerationState) -> Optional[WorkflowStep]:
    """
    Get information about the current workflow step.

    Args:
        state: The workflow state

    Returns:
        Current step information or None
    """
    current_step_name = state["current_step"]
    for step in reversed(state["step_history"]):
        if step.name == current_step_name:
            return step
    return None


def is_step_completed(state: CVGenerationState, step_name: str) -> bool:
    """
    Check if a specific step has been completed.

    Args:
        state: The workflow state
        step_name: Name of the step to check

    Returns:
        True if step is completed, False otherwise
    """
    for step in state["step_history"]:
        if step.name == step_name and step.status == "completed":
            return True
    return False


def get_failed_steps(state: CVGenerationState) -> List[WorkflowStep]:
    """
    Get all failed steps from the workflow.

    Args:
        state: The workflow state

    Returns:
        List of failed workflow steps
    """
    return [step for step in state["step_history"] if step.status == "failed"]


def can_retry_step(state: CVGenerationState, step_name: str) -> bool:
    """
    Check if a step can be retried based on retry limits.

    Args:
        state: The workflow state
        step_name: Name of the step to check

    Returns:
        True if step can be retried, False otherwise
    """
    retry_count = state["retry_attempts"].get(step_name, 0)
    return retry_count < state["max_retries"]