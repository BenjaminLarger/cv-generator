"""
LangGraph workflow for CV/Cover Letter generation.

This module implements the complete workflow orchestration using LangGraph,
including all nodes, conditional routing, human-in-the-loop processes,
and error handling for the CV generation pipeline.
"""

import asyncio
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from agents.pdf_generator import PDFGenerator, PDFGeneratorError
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents.template_customizer import TemplateCustomizer
from agents.preview_generator import PreviewGenerator, PreviewGeneratorError
from .workflow_state import (
    CVGenerationState, WorkflowStep, HumanApprovalRequest,
    create_initial_state, add_log_entry, update_state_progress,
    is_step_completed, can_retry_step
)
from .state_manager import StateManager, StateManagerError
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.job_analyzer import JobAnalyzer
from agents.profile_matcher import ProfileMatcher, ProfileMatchingError
from models.user_profile import UserProfile
from models.job_data import JobData
from models.match_result import MatchResult
from utils.yaml_parser import YAMLParser
from utils.file_manager import FileManager, organize_output_files, cleanup_temp_files
from utils.logging_config import get_logger

logger = get_logger(__name__)


class WorkflowError(Exception):
    """Base exception for workflow errors."""
    pass


class NodeExecutionError(WorkflowError):
    """Exception raised when a workflow node fails."""
    pass


class WorkflowInterruptedError(WorkflowError):
    """Exception raised when workflow is interrupted."""
    pass


class CVGenerationWorkflow:
    """
    Complete LangGraph workflow for CV/Cover Letter generation.

    This class orchestrates the entire CV generation pipeline using LangGraph,
    providing robust error handling, human-in-the-loop processes, state
    persistence, and progress tracking.

    Features:
    - LangGraph-based workflow orchestration
    - Human approval checkpoints
    - Automatic error recovery and retry logic
    - State persistence and resume capability
    - Interactive and batch processing modes
    - Comprehensive progress tracking
    - Graceful error handling and rollback
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state_manager: Optional[StateManager] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the CV generation workflow.

        Args:
            config: Optional workflow configuration
            state_manager: Optional state manager instance
            debug_mode: Whether to enable debug mode
        """
        self.config = config or {}
        self.debug_mode = debug_mode
        self.state_manager = state_manager

        # Initialize agents
        self._initialize_agents()

        # Create workflow graph
        self.workflow_graph = self._create_workflow_graph()

        # Compile the graph
        self.compiled_workflow = self.workflow_graph.compile(
            checkpointer=MemorySaver() if not state_manager else None
        )

        logger.info("CVGenerationWorkflow initialized successfully")

    def _initialize_agents(self) -> None:
        """Initialize all workflow agents."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise WorkflowError("OPENAI_API_KEY environment variable not set")

            self.job_analyzer = JobAnalyzer(
                api_key=api_key,
                model=self.config.get('openai_model', 'gpt-4'),
                max_retries=self.config.get('max_retries', 3)
            )

            self.profile_matcher = ProfileMatcher(
                api_key=api_key,
                model=self.config.get('openai_model', 'gpt-4'),
                max_retries=self.config.get('max_retries', 3)
            )

            self.template_customizer = TemplateCustomizer(
                templates_dir=self.config.get('templates_dir'),
            )

            self.preview_generator = PreviewGenerator()

            self.pdf_generator = PDFGenerator()

            logger.info("All workflow agents initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize workflow agents: {e}"
            logger.error(error_msg)
            raise WorkflowError(error_msg) from e

    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow graph."""
        # Create the state graph
        workflow = StateGraph(CVGenerationState)

        # Add all workflow nodes
        workflow.add_node("load_user_profile", self.load_user_profile_node)
        workflow.add_node("analyze_job", self.analyze_job_node)
        workflow.add_node("match_profile", self.match_profile_node)
        workflow.add_node("customize_templates", self.customize_templates_node)
        workflow.add_node("generate_preview", self.generate_preview_node)
        workflow.add_node("human_approval", self.human_approval_node)
        workflow.add_node("generate_pdfs", self.generate_pdfs_node)
        workflow.add_node("cleanup", self.cleanup_node)

        # Add error handling node
        workflow.add_node("handle_error", self.handle_error_node)

        # Define the workflow edges
        workflow.add_edge(START, "load_user_profile")
        workflow.add_edge("load_user_profile", "analyze_job")
        workflow.add_edge("analyze_job", "match_profile")
        workflow.add_edge("match_profile", "customize_templates")
        workflow.add_edge("customize_templates", "generate_preview")

        # Conditional edge for human approval
        workflow.add_conditional_edges(
            "generate_preview",
            self.should_request_approval,
            {
                "request_approval": "human_approval",
                "skip_approval": "generate_pdfs"
            }
        )

        # Conditional edge after approval
        workflow.add_conditional_edges(
            "human_approval",
            self.check_approval_response,
            {
                "approved": "generate_pdfs",
                "rejected": "customize_templates",
                "retry": "generate_preview"
            }
        )

        workflow.add_edge("generate_pdfs", "cleanup")
        workflow.add_edge("cleanup", END)

        # Error handling edges
        workflow.add_conditional_edges(
            "handle_error",
            self.should_retry_or_fail,
            {
                "retry": "load_user_profile",  # Start from beginning for safety
                "fail": END
            }
        )

        return workflow

    async def run_workflow(
        self,
        job_input: str,
        user_profile_path: str,
        output_directory: str,
        interactive_mode: bool = True,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete CV generation workflow.

        Args:
            job_input: Job URL or raw text
            user_profile_path: Path to user profile YAML
            output_directory: Output directory path
            interactive_mode: Whether to run in interactive mode
            resume_from_checkpoint: Optional checkpoint to resume from

        Returns:
            Dictionary containing workflow results

        Raises:
            WorkflowError: If workflow execution fails
        """
        try:
            # Create or load state
            if resume_from_checkpoint and self.state_manager:
                logger.info(f"Resuming workflow from checkpoint: {resume_from_checkpoint}")
                state = self.state_manager.load_state(resume_from_checkpoint)
                state["interactive_mode"] = interactive_mode
            else:
                logger.info("Starting new workflow")
                state = create_initial_state(
                    job_input=job_input,
                    user_profile_path=user_profile_path,
                    output_directory=output_directory,
                    config=self.config,
                    interactive_mode=interactive_mode,
                    debug_mode=self.debug_mode
                )

            # Update workflow status
            state["workflow_status"] = "running"
            add_log_entry(state, "INFO", "Workflow execution started")

            # Save initial state if state manager available
            if self.state_manager:
                self.state_manager.save_state(state, "initial")

            # Execute the workflow
            config = {"configurable": {"thread_id": state["workflow_id"]}}

            # Run the compiled workflow
            result = await self.compiled_workflow.ainvoke(state, config=config)

            # Update final status
            result["workflow_status"] = "completed"
            result["progress_percentage"] = 100.0
            add_log_entry(result, "INFO", "Workflow execution completed successfully")

            # Save final state
            if self.state_manager:
                self.state_manager.save_state(result, "completed")

            logger.info("Workflow completed successfully")
            return self._create_workflow_result(result)

        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            logger.error(error_msg)
            if self.debug_mode:
                logger.error(f"Full traceback: {traceback.format_exc()}")

            # Update state with error
            if 'state' in locals():
                state["workflow_status"] = "failed"
                state["errors"].append(error_msg)
                add_log_entry(state, "ERROR", error_msg)

                # Save error state
                if self.state_manager:
                    try:
                        self.state_manager.save_state(state, "failed")
                    except StateManagerError:
                        logger.warning("Failed to save error state")

            raise WorkflowError(error_msg) from e

    # Workflow Nodes

    async def load_user_profile_node(self, state: CVGenerationState) -> CVGenerationState:
        """Load and validate user profile from YAML file."""
        step_name = "load_user_profile"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Loading user profile", step_name)

            profile_path = Path(state["user_profile_path"])

            if not profile_path.exists():
                raise FileNotFoundError(f"User profile file not found: {profile_path}")

            # Load profile using YAMLParser
            user_profile = YAMLParser.load_user_profile(str(profile_path))

            # Validate profile
            if not user_profile.personal_info.name:
                raise ValueError("User profile must have a name")

            if not user_profile.personal_info.email:
                raise ValueError("User profile must have an email")

            # Update state
            state["user_profile"] = user_profile
            state["current_step"] = "analyze_job"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"User profile loaded successfully: {user_profile.personal_info.name}",
                step_name,
                {"skills_count": len(user_profile.skills), "experiences_count": len(user_profile.experiences)}
            )

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except Exception as e:
            error_msg = f"Failed to load user profile: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)

            if can_retry_step(state, step_name):
                state["retry_attempts"][step_name] = state["retry_attempts"].get(step_name, 0) + 1
                add_log_entry(state, "WARNING", f"Retrying step {step_name}", step_name)
                return await self.load_user_profile_node(state)
            else:
                state["workflow_status"] = "failed"
                raise NodeExecutionError(error_msg) from e

    async def analyze_job_node(self, state: CVGenerationState) -> CVGenerationState:
        """Analyze job posting and extract structured data."""
        step_name = "analyze_job"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Analyzing job posting", step_name)

            # Analyze job using JobAnalyzer
            job_data = self.job_analyzer.analyze_job(state["job_input"])

            # Update state
            state["job_data"] = job_data
            state["current_step"] = "match_profile"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"Job analysis completed: {job_data.position} at {job_data.company_name}",
                step_name,
                {
                    "skills_required": len(job_data.skills_required),
                    "requirements": len(job_data.requirements),
                    "experience_level": job_data.experience_level
                }
            )

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except JobAnalysisError as e:
            error_msg = f"Job analysis failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)

            if can_retry_step(state, step_name):
                state["retry_attempts"][step_name] = state["retry_attempts"].get(step_name, 0) + 1
                add_log_entry(state, "WARNING", f"Retrying step {step_name}", step_name)
                return await self.analyze_job_node(state)
            else:
                state["workflow_status"] = "failed"
                raise NodeExecutionError(error_msg) from e

    async def match_profile_node(self, state: CVGenerationState) -> CVGenerationState:
        """Match user profile against job requirements."""
        step_name = "match_profile"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Matching profile against job requirements", step_name)

            if not state["user_profile"] or not state["job_data"]:
                raise ValueError("User profile and job data must be loaded first")

            # Perform matching using ProfileMatcher
            match_result = self.profile_matcher.match_profile(
                state["job_data"],
                state["user_profile"]
            )

            # Update state
            state["match_result"] = match_result
            state["current_step"] = "customize_templates"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"Profile matching completed: {match_result.get_match_category()} (Score: {match_result.score}/10)",
                step_name,
                {
                    "match_score": match_result.score,
                    "matched_skills": len(match_result.matched_skills),
                    "relevant_experiences": len(match_result.relevant_experiences),
                    "suggestions": len(match_result.suggestions)
                }
            )

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except ProfileMatchingError as e:
            error_msg = f"Profile matching failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)

            if can_retry_step(state, step_name):
                state["retry_attempts"][step_name] = state["retry_attempts"].get(step_name, 0) + 1
                add_log_entry(state, "WARNING", f"Retrying step {step_name}", step_name)
                return await self.match_profile_node(state)
            else:
                state["workflow_status"] = "failed"
                raise NodeExecutionError(error_msg) from e

    async def customize_templates_node(self, state: CVGenerationState) -> CVGenerationState:
        """Customize CV and cover letter templates."""
        step_name = "customize_templates"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Customizing templates", step_name)

            if not all([state["user_profile"], state["job_data"], state["match_result"]]):
                raise ValueError("User profile, job data, and match result must be available")

            # Customize templates using TemplateCustomizer
            customization_result = self.template_customizer.customize_templates(
                match_result=state["match_result"],
                user_profile=state["user_profile"],
                job_data=state["job_data"]
            )

            # Update state
            state["customized_templates"] = {
                "cv_html": customization_result.cv_html,
                "cover_letter_html": customization_result.cover_letter_html
            }
            state["current_step"] = "generate_preview"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"Template customization completed: {len(customization_result.changes_made)} changes made",
                step_name,
                {
                    "changes_count": len(customization_result.changes_made),
                    "customization_score": customization_result.customization_score,
                    "changes": customization_result.changes_made[:5]  # First 5 changes
                }
            )

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except Exception as e:
            error_msg = f"Template customization failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)

            if can_retry_step(state, step_name):
                state["retry_attempts"][step_name] = state["retry_attempts"].get(step_name, 0) + 1
                add_log_entry(state, "WARNING", f"Retrying step {step_name}", step_name)
                return await self.customize_templates_node(state)
            else:
                state["workflow_status"] = "failed"
                raise NodeExecutionError(error_msg) from e

    async def generate_preview_node(self, state: CVGenerationState) -> CVGenerationState:
        """Generate preview content for user review."""
        step_name = "generate_preview"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Generating preview content", step_name)

            if not all([state["customized_templates"], state["job_data"], state["match_result"]]):
                raise ValueError("Customized templates, job data, and match result must be available")

            # Generate comprehensive preview using PreviewGenerator
            customized_html = {
                "cv": state["customized_templates"]["cv_html"],
                "cover_letter": state["customized_templates"]["cover_letter_html"]
            }

            # Get changes from template customization
            changes = state.get("customization_changes", ["Template customized for job requirements"])

            preview_result = self.preview_generator.generate_preview(
                customized_html=customized_html,
                match_result=state["match_result"],
                changes=changes
            )

            state["preview_content"] = {
                "interactive_html": preview_result,
                "cv_html": state["customized_templates"]["cv_html"],
                "cover_letter_html": state["customized_templates"]["cover_letter_html"],
                "match_score": state["match_result"].score
            }

            state["current_step"] = "human_approval"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"Preview content generated successfully: {len(preview_result)} chars",
                step_name,
                {"has_interactive": bool(preview_result)}
            )

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except PreviewGeneratorError as e:
            error_msg = f"Preview generation failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)

            if can_retry_step(state, step_name):
                state["retry_attempts"][step_name] = state["retry_attempts"].get(step_name, 0) + 1
                add_log_entry(state, "WARNING", f"Retrying step {step_name}", step_name)
                return await self.generate_preview_node(state)
            else:
                state["workflow_status"] = "failed"
                raise NodeExecutionError(error_msg) from e

    async def human_approval_node(self, state: CVGenerationState) -> CVGenerationState:
        """Handle human approval process."""
        step_name = "human_approval"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Requesting human approval", step_name)

            if not state["preview_content"]:
                raise ValueError("Preview content must be available")

            # Create approval request
            approval_request = HumanApprovalRequest(
                step_name=step_name,
                content_preview=f"Match Score: {state['preview_content']['match_score']}\n\n"
                              f"CV HTML Length: {len(state['preview_content']['cv_html'])} chars\n\n"
                              f"Cover Letter HTML Length: {len(state['preview_content']['cover_letter_html'])} chars\n\n"
                              f"Interactive Preview Available: Yes",
                approval_type="preview",
                options=["approve", "reject", "modify"],
                context={
                    "match_score": state["match_result"].score if state["match_result"] else 0,
                    "job_title": state["job_data"].position if state["job_data"] else "Unknown"
                }
            )

            state["approval_request"] = approval_request
            state["workflow_status"] = "paused"

            # In interactive mode, wait for user input
            if state["interactive_mode"]:
                approval_response = await self._get_user_approval(approval_request)
                state["approval_response"] = approval_response

            state["current_step"] = "generate_pdfs"  # Will be updated based on approval

            step.complete()
            state["step_history"].append(step)

            add_log_entry(state, "INFO", "Approval request processed", step_name)

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except Exception as e:
            error_msg = f"Human approval failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)
            state["workflow_status"] = "failed"
            raise NodeExecutionError(error_msg) from e

    async def generate_pdfs_node(self, state: CVGenerationState) -> CVGenerationState:
        """Generate final PDF documents."""
        step_name = "generate_pdfs"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Generating PDF documents", step_name)

            if not all([state["customized_templates"], state["job_data"]]):
                raise ValueError("Customized templates and job data must be available")

            # Generate PDFs using PDFGenerator
            approved_html = {
                "cv": state["customized_templates"]["cv_html"],
                "cover_letter": state["customized_templates"]["cover_letter_html"]
            }

            pdf_result = await self.pdf_generator.generate_pdfs(
                approved_html=approved_html,
                job_data=state["job_data"]
            )

            state["final_pdfs"] = {
                "cv_path": pdf_result["cv"],
                "cover_letter_path": pdf_result["cover_letter"]
            }

            # Add to output files list
            output_files = [pdf_result["cv"], pdf_result["cover_letter"]]
            if "preview_html" in pdf_result:
                output_files.append(pdf_result["preview_html"])

            state["output_files"].extend(output_files)
            state["current_step"] = "cleanup"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"PDF documents generated successfully",
                step_name,
                {
                    "cv_path": pdf_result["cv"],
                    "cover_letter_path": pdf_result["cover_letter"]
                }
            )

            # Save checkpoint
            if self.state_manager:
                self.state_manager.save_state(state, step_name)

            return state

        except PDFGeneratorError as e:
            error_msg = f"PDF generation failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)

            if can_retry_step(state, step_name):
                state["retry_attempts"][step_name] = state["retry_attempts"].get(step_name, 0) + 1
                add_log_entry(state, "WARNING", f"Retrying step {step_name}", step_name)
                return await self.generate_pdfs_node(state)
            else:
                state["workflow_status"] = "failed"
                raise NodeExecutionError(error_msg) from e

    async def cleanup_node(self, state: CVGenerationState) -> CVGenerationState:
        """Clean up temporary files and organize output."""
        step_name = "cleanup"
        step = WorkflowStep(name=step_name)
        step.start()

        try:
            add_log_entry(state, "INFO", "Cleaning up and organizing files", step_name)

            # Organize output files
            if state["output_files"]:
                organize_output_files(state["output_files"], state["output_directory"])

            # Clean up temporary files
            if state["temp_files"]:
                cleanup_temp_files(state["temp_files"])

            # Update progress
            update_state_progress(state)
            state["workflow_status"] = "completed"
            state["current_step"] = "completed"

            step.complete()
            state["step_history"].append(step)

            add_log_entry(
                state, "INFO",
                f"Workflow cleanup completed successfully",
                step_name,
                {"output_files": len(state["output_files"]), "temp_files_cleaned": len(state["temp_files"])}
            )

            return state

        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            step.fail(error_msg)
            state["step_history"].append(step)
            state["errors"].append(error_msg)
            add_log_entry(state, "ERROR", error_msg, step_name)
            # Don't fail the workflow for cleanup errors
            logger.warning(error_msg)
            return state

    async def handle_error_node(self, state: CVGenerationState) -> CVGenerationState:
        """Handle workflow errors and determine recovery strategy."""
        step_name = "handle_error"
        add_log_entry(state, "INFO", "Handling workflow error", step_name)

        # Update workflow status
        state["workflow_status"] = "failed"
        state["current_step"] = "error_recovery"

        # Log error details
        if state["errors"]:
            latest_error = state["errors"][-1]
            add_log_entry(state, "ERROR", f"Workflow failed with error: {latest_error}", step_name)

        return state

    # Conditional Edge Functions

    def should_request_approval(self, state: CVGenerationState) -> str:
        """Determine if human approval should be requested."""
        if state["interactive_mode"]:
            return "request_approval"
        else:
            return "skip_approval"

    def check_approval_response(self, state: CVGenerationState) -> str:
        """Check the human approval response."""
        if not state.get("approval_response"):
            return "retry"  # No response yet

        response = state["approval_response"]
        action = response.get("action", "retry")

        if action == "approve":
            return "approved"
        elif action == "reject":
            return "rejected"
        else:
            return "retry"

    def should_retry_or_fail(self, state: CVGenerationState) -> str:
        """Determine if workflow should retry or fail."""
        # Check if any steps can be retried
        for step in state["step_history"]:
            if step.status == "failed" and can_retry_step(state, step.name):
                return "retry"

        return "fail"

    # Helper Methods

    async def _get_user_approval(self, approval_request: HumanApprovalRequest) -> Dict[str, Any]:
        """Get user approval in interactive mode."""
        print("\n" + "="*60)
        print("🔍 WORKFLOW APPROVAL REQUIRED")
        print("="*60)
        print(f"\nStep: {approval_request.step_name}")
        print(f"Type: {approval_request.approval_type}")
        print(f"\nContent Preview:")
        print("-" * 40)
        print(approval_request.content_preview)
        print("-" * 40)

        while True:
            print(f"\nAvailable options: {', '.join(approval_request.options)}")
            user_input = input("\nYour choice (approve/reject/modify): ").strip().lower()

            if user_input in approval_request.options:
                return {
                    "action": user_input,
                    "timestamp": datetime.now().isoformat(),
                    "comments": input("Optional comments: ").strip() or None
                }
            else:
                print(f"Invalid option. Please choose from: {', '.join(approval_request.options)}")

    def _create_workflow_result(self, state: CVGenerationState) -> Dict[str, Any]:
        """Create final workflow result summary."""
        return {
            "workflow_id": state["workflow_id"],
            "status": state["workflow_status"],
            "progress": state["progress_percentage"],
            "duration_seconds": (datetime.now() - state["start_time"]).total_seconds(),
            "output_files": state.get("final_pdfs", {}),
            "match_score": state["match_result"].score if state["match_result"] else None,
            "errors": state["errors"],
            "warnings": state["warnings"],
            "steps_completed": len([s for s in state["step_history"] if s.status == "completed"]),
            "total_steps": len(state["step_history"])
        }

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status."""
        if not self.state_manager:
            return None

        try:
            checkpoints = self.state_manager.list_checkpoints(workflow_id)
            if not checkpoints:
                return None

            latest = checkpoints[0]
            return {
                "workflow_id": workflow_id,
                "current_step": latest.step_name,
                "timestamp": latest.timestamp.isoformat(),
                "progress": latest.metadata.get("progress", 0),
                "status": latest.metadata.get("status", "unknown")
            }

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None