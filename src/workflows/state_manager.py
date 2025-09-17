"""
State management for CV generation workflow.

This module provides comprehensive state persistence, recovery, and management
functionality for the LangGraph workflow, including checkpoint creation,
state serialization, and error recovery mechanisms.
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import pickle
import gzip

from pydantic import BaseModel, Field, ValidationError

from .workflow_state import CVGenerationState, WorkflowStep, add_log_entry
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger

logger = get_logger(__name__)


class PathAwareJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Path objects by converting them to strings."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)


class StateManagerError(Exception):
    """Base exception for state manager errors."""
    pass


class StateSerializationError(StateManagerError):
    """Exception raised when state serialization fails."""
    pass


class StateDeserializationError(StateManagerError):
    """Exception raised when state deserialization fails."""
    pass


class StateValidationError(StateManagerError):
    """Exception raised when state validation fails."""
    pass


@dataclass
class WorkflowCheckpoint:
    """Model for workflow checkpoints."""

    workflow_id: str
    step_name: str
    timestamp: datetime
    state_snapshot: Dict[str, Any]
    metadata: Dict[str, Any]
    file_size_bytes: int
    compressed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        """Create checkpoint from dictionary."""
        # Handle datetime conversion
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class WorkflowState(BaseModel):
    """Pydantic model for workflow state validation."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    current_step: str = Field(..., description="Current workflow step")
    workflow_status: str = Field(..., description="Workflow status")
    start_time: datetime = Field(..., description="Workflow start time")
    last_update: datetime = Field(..., description="Last update time")
    progress_percentage: float = Field(ge=0, le=100, description="Progress percentage")
    state_version: str = Field(..., description="State schema version")

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class StateManager:
    """
    Comprehensive state management for CV generation workflows.

    This class handles state persistence, recovery, validation, and cleanup
    for the LangGraph workflow system. It supports both JSON and binary
    serialization with compression and provides robust error handling.

    Features:
    - Automatic checkpoint creation at key workflow steps
    - State compression and efficient storage
    - Version compatibility checking
    - Automatic cleanup of old state files
    - Error recovery and state rollback
    - Performance metrics tracking
    """

    def __init__(
        self,
        state_directory: Union[str, Path],
        max_checkpoints: int = 10,
        compress_states: bool = True,
        auto_cleanup: bool = True,
        cleanup_age_days: int = 7
    ):
        """
        Initialize the StateManager.

        Args:
            state_directory: Directory to store state files
            max_checkpoints: Maximum number of checkpoints to keep per workflow
            compress_states: Whether to compress state files
            auto_cleanup: Whether to automatically clean up old files
            cleanup_age_days: Age in days after which to clean up state files
        """
        self.state_directory = Path(state_directory)
        self.max_checkpoints = max_checkpoints
        self.compress_states = compress_states
        self.auto_cleanup = auto_cleanup
        self.cleanup_age_days = cleanup_age_days

        # Create state directory if it doesn't exist
        self.state_directory.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoints_dir = self.state_directory / "checkpoints"
        self.backups_dir = self.state_directory / "backups"
        self.temp_dir = self.state_directory / "temp"

        for directory in [self.checkpoints_dir, self.backups_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)

        logger.info(f"StateManager initialized with directory: {self.state_directory}")

    def save_state(
        self,
        state: CVGenerationState,
        checkpoint_name: Optional[str] = None,
        create_backup: bool = True
    ) -> str:
        """
        Save workflow state to disk.

        Args:
            state: The workflow state to save
            checkpoint_name: Optional checkpoint name (defaults to current step)
            create_backup: Whether to create a backup of existing state

        Returns:
            Path to the saved state file

        Raises:
            StateSerializationError: If state saving fails
        """
        try:
            workflow_id = state["workflow_id"]
            checkpoint_name = checkpoint_name or state["current_step"]
            timestamp = datetime.now()

            # Create filename
            filename = f"{workflow_id}_{checkpoint_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

            if self.compress_states:
                filepath = self.checkpoints_dir / f"{filename}.json.gz"
            else:
                filepath = self.checkpoints_dir / f"{filename}.json"

            # Create backup if requested and file exists
            if create_backup and filepath.exists():
                self._create_backup(filepath)

            # Prepare state for serialization
            serializable_state = self._prepare_state_for_serialization(state)

            # Create checkpoint metadata
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_id,
                step_name=checkpoint_name,
                timestamp=timestamp,
                state_snapshot=serializable_state,
                metadata={
                    "total_steps": len(state["step_history"]),
                    "errors_count": len(state["errors"]),
                    "progress": state["progress_percentage"],
                    "interactive_mode": state["interactive_mode"]
                },
                file_size_bytes=0,  # Will be updated after saving
                compressed=self.compress_states
            )

            # Save to file
            self._write_checkpoint_file(checkpoint, filepath)

            # Update file size
            checkpoint.file_size_bytes = filepath.stat().st_size

            # Update state with backup path
            state["backup_state_path"] = str(filepath)

            # Add log entry
            add_log_entry(
                state,
                "INFO",
                f"State saved to checkpoint: {filepath.name}",
                checkpoint_name,
                {"file_size": checkpoint.file_size_bytes, "compressed": self.compress_states}
            )

            # Cleanup old checkpoints
            if self.auto_cleanup:
                self._cleanup_old_checkpoints(workflow_id)

            logger.info(f"State saved successfully: {filepath}")
            return str(filepath)

        except Exception as e:
            error_msg = f"Failed to save state: {e}"
            logger.error(error_msg)
            raise StateSerializationError(error_msg) from e

    def load_state(self, filepath: Union[str, Path]) -> CVGenerationState:
        """
        Load workflow state from disk.

        Args:
            filepath: Path to the state file

        Returns:
            Loaded workflow state

        Raises:
            StateDeserializationError: If state loading fails
        """
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                raise StateDeserializationError(f"State file not found: {filepath}")

            logger.info(f"Loading state from: {filepath}")

            # Load checkpoint
            checkpoint = self._read_checkpoint_file(filepath)

            # Validate state structure
            self._validate_state_structure(checkpoint.state_snapshot)

            # Convert back to proper types
            state = self._prepare_state_from_serialization(checkpoint.state_snapshot)

            # Update state with load information
            add_log_entry(
                state,
                "INFO",
                f"State loaded from checkpoint: {filepath.name}",
                state["current_step"],
                {"checkpoint_age": (datetime.now() - checkpoint.timestamp).total_seconds()}
            )

            logger.info(f"State loaded successfully from: {filepath}")
            return state

        except Exception as e:
            error_msg = f"Failed to load state: {e}"
            logger.error(error_msg)
            raise StateDeserializationError(error_msg) from e

    def list_checkpoints(self, workflow_id: Optional[str] = None) -> List[WorkflowCheckpoint]:
        """
        List available checkpoints.

        Args:
            workflow_id: Optional workflow ID to filter by

        Returns:
            List of available checkpoints
        """
        checkpoints = []

        try:
            for filepath in self.checkpoints_dir.glob("*.json*"):
                try:
                    checkpoint = self._read_checkpoint_file(filepath)

                    if workflow_id is None or checkpoint.workflow_id == workflow_id:
                        checkpoints.append(checkpoint)

                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {filepath}: {e}")
                    continue

            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)

            logger.debug(f"Found {len(checkpoints)} checkpoints")
            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def get_latest_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Get the latest checkpoint for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Latest checkpoint or None if not found
        """
        checkpoints = self.list_checkpoints(workflow_id)
        return checkpoints[0] if checkpoints else None

    def delete_checkpoint(self, filepath: Union[str, Path]) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            filepath = Path(filepath)

            if filepath.exists():
                # Move to backup before deletion
                backup_path = self.backups_dir / f"deleted_{filepath.name}"
                shutil.move(str(filepath), str(backup_path))

                logger.info(f"Checkpoint deleted (backed up): {filepath}")
                return True
            else:
                logger.warning(f"Checkpoint file not found: {filepath}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {filepath}: {e}")
            return False

    def cleanup_old_checkpoints(self, workflow_id: Optional[str] = None) -> int:
        """
        Clean up old checkpoint files.

        Args:
            workflow_id: Optional workflow ID to limit cleanup

        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.cleanup_age_days)

        try:
            for filepath in self.checkpoints_dir.glob("*.json*"):
                try:
                    # Check file age
                    file_time = datetime.fromtimestamp(filepath.stat().st_mtime)

                    if file_time < cutoff_date:
                        # If workflow_id specified, check if it matches
                        if workflow_id:
                            checkpoint = self._read_checkpoint_file(filepath)
                            if checkpoint.workflow_id != workflow_id:
                                continue

                        # Move to backup
                        backup_path = self.backups_dir / f"cleanup_{filepath.name}"
                        shutil.move(str(filepath), str(backup_path))
                        cleaned_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process file {filepath} during cleanup: {e}")
                    continue

            logger.info(f"Cleaned up {cleaned_count} old checkpoint files")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed during checkpoint cleanup: {e}")
            return 0

    def validate_state(self, state: CVGenerationState) -> bool:
        """
        Validate workflow state structure and data integrity.

        Args:
            state: The workflow state to validate

        Returns:
            True if state is valid, False otherwise

        Raises:
            StateValidationError: If critical validation errors are found
        """
        try:
            # Create WorkflowState model for validation
            workflow_state = WorkflowState(
                workflow_id=state["workflow_id"],
                current_step=state["current_step"],
                workflow_status=state["workflow_status"],
                start_time=state["start_time"],
                last_update=state["last_update"],
                progress_percentage=state["progress_percentage"],
                state_version=state["state_version"]
            )

            # Check required fields
            required_fields = [
                "job_input", "user_profile_path", "output_directory",
                "current_step", "workflow_status", "start_time"
            ]

            for field in required_fields:
                if field not in state or state[field] is None:
                    raise StateValidationError(f"Required field missing: {field}")

            # Validate step history
            for step in state["step_history"]:
                if not isinstance(step, WorkflowStep):
                    # Try to convert if it's a dict
                    if isinstance(step, dict):
                        try:
                            WorkflowStep(**step)
                        except Exception as e:
                            raise StateValidationError(f"Invalid step in history: {e}")
                    else:
                        raise StateValidationError(f"Invalid step type in history: {type(step)}")

            # Validate state version compatibility
            if state["state_version"] != "1.0":
                logger.warning(f"State version mismatch: {state['state_version']} (expected 1.0)")

            logger.debug("State validation passed")
            return True

        except ValidationError as e:
            raise StateValidationError(f"Pydantic validation failed: {e}")
        except Exception as e:
            raise StateValidationError(f"State validation failed: {e}")

    def _prepare_state_for_serialization(self, state: CVGenerationState) -> Dict[str, Any]:
        """Prepare state for JSON serialization."""
        serializable = dict(state)

        # Convert datetime objects
        datetime_fields = ["start_time", "last_update", "estimated_completion"]
        for field in datetime_fields:
            if field in serializable and serializable[field]:
                serializable[field] = serializable[field].isoformat()

        # Convert Pydantic models to dictionaries
        if serializable.get("user_profile"):
            serializable["user_profile"] = serializable["user_profile"].to_dict()

        if serializable.get("job_data"):
            serializable["job_data"] = serializable["job_data"].to_dict()

        if serializable.get("match_result"):
            serializable["match_result"] = serializable["match_result"].to_dict()

        # Convert WorkflowStep objects to dictionaries
        step_history = []
        for step in serializable.get("step_history", []):
            if isinstance(step, WorkflowStep):
                step_dict = step.dict()
                # Convert datetime fields in steps
                for dt_field in ["start_time", "end_time"]:
                    if step_dict.get(dt_field):
                        step_dict[dt_field] = step_dict[dt_field].isoformat()
                step_history.append(step_dict)
            else:
                step_history.append(step)
        serializable["step_history"] = step_history

        # Convert approval request if present
        if serializable.get("approval_request"):
            approval_req = serializable["approval_request"]
            if hasattr(approval_req, 'dict'):
                approval_dict = approval_req.dict()
                if approval_dict.get("requested_at"):
                    approval_dict["requested_at"] = approval_dict["requested_at"].isoformat()
                serializable["approval_request"] = approval_dict

        return serializable

    def _prepare_state_from_serialization(self, data: Dict[str, Any]) -> CVGenerationState:
        """Convert serialized data back to proper state types."""
        from ..models.user_profile import UserProfile
        from ..models.job_data import JobData
        from ..models.match_result import MatchResult
        from .workflow_state import HumanApprovalRequest

        # Convert datetime strings back to datetime objects
        datetime_fields = ["start_time", "last_update", "estimated_completion"]
        for field in datetime_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])

        # Convert model dictionaries back to Pydantic models
        if data.get("user_profile"):
            data["user_profile"] = UserProfile.from_dict(data["user_profile"])

        if data.get("job_data"):
            data["job_data"] = JobData.from_dict(data["job_data"])

        if data.get("match_result"):
            data["match_result"] = MatchResult.from_dict(data["match_result"])

        # Convert step dictionaries back to WorkflowStep objects
        step_history = []
        for step_data in data.get("step_history", []):
            if isinstance(step_data, dict):
                # Convert datetime fields
                for dt_field in ["start_time", "end_time"]:
                    if step_data.get(dt_field):
                        step_data[dt_field] = datetime.fromisoformat(step_data[dt_field])
                step_history.append(WorkflowStep(**step_data))
            else:
                step_history.append(step_data)
        data["step_history"] = step_history

        # Convert approval request
        if data.get("approval_request"):
            approval_data = data["approval_request"]
            if isinstance(approval_data, dict):
                if approval_data.get("requested_at"):
                    approval_data["requested_at"] = datetime.fromisoformat(approval_data["requested_at"])
                data["approval_request"] = HumanApprovalRequest(**approval_data)

        return data  # type: ignore

    def _write_checkpoint_file(self, checkpoint: WorkflowCheckpoint, filepath: Path) -> None:
        """Write checkpoint to file with optional compression."""
        checkpoint_dict = checkpoint.to_dict()

        # Convert datetime to string for serialization
        checkpoint_dict["timestamp"] = checkpoint_dict["timestamp"].isoformat()

        if self.compress_states:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(checkpoint_dict, f, indent=2, ensure_ascii=False, cls=PathAwareJSONEncoder)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_dict, f, indent=2, ensure_ascii=False, cls=PathAwareJSONEncoder)

    def _read_checkpoint_file(self, filepath: Path) -> WorkflowCheckpoint:
        """Read checkpoint from file with automatic decompression."""
        try:
            # Try compressed first
            if filepath.suffix == '.gz' or self.compress_states:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            return WorkflowCheckpoint.from_dict(data)

        except Exception as e:
            # Try uncompressed if compressed failed
            if filepath.suffix == '.gz':
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return WorkflowCheckpoint.from_dict(data)
                except:
                    pass
            raise e

    def _validate_state_structure(self, state_data: Dict[str, Any]) -> None:
        """Validate basic state structure."""
        required_keys = [
            "workflow_id", "current_step", "workflow_status",
            "start_time", "state_version"
        ]

        for key in required_keys:
            if key not in state_data:
                raise StateValidationError(f"Missing required state key: {key}")

    def _create_backup(self, filepath: Path) -> None:
        """Create backup of existing file."""
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.name}"
        backup_path = self.backups_dir / backup_name
        shutil.copy2(filepath, backup_path)

    def _cleanup_old_checkpoints(self, workflow_id: str) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = self.list_checkpoints(workflow_id)

        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            to_remove = checkpoints[self.max_checkpoints:]

            for checkpoint in to_remove:
                # Find the file path
                for filepath in self.checkpoints_dir.glob("*.json*"):
                    try:
                        file_checkpoint = self._read_checkpoint_file(filepath)
                        if (file_checkpoint.workflow_id == checkpoint.workflow_id and
                            file_checkpoint.timestamp == checkpoint.timestamp):
                            self.delete_checkpoint(filepath)
                            break
                    except:
                        continue