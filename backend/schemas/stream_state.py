"""
Stream State Schema
Core data model for pipeline state tracking and real-time streaming.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class StageEnum(str, Enum):
    """Pipeline stage enumeration."""
    INPUT = "input"
    PREPROCESSING = "preprocessing"
    TASK_DETECTION = "task_detection"
    TOOL_SELECTION = "tool_selection"
    EXECUTION = "execution"
    AGGREGATION = "aggregation"
    DRAFT_OUTPUT = "draft_output"
    FINAL_OUTPUT = "final_output"


class StatusEnum(str, Enum):
    """Pipeline status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"
    WAITING_APPROVAL = "waiting_approval"


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DECISION = "decision"  # Special level for glass-box decisions


class LogEntry(BaseModel):
    """Individual log entry with timestamp and metadata."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel = LogLevel.INFO
    stage: StageEnum
    message: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StageMetadata(BaseModel):
    """Stage-specific metadata for transparency."""
    stage: StageEnum
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    decisions_made: List[str] = Field(default_factory=list)
    intermediate_data: Optional[Dict[str, Any]] = None


class StreamState(BaseModel):
    """
    Central state object for the entire pipeline.
    This is streamed to the dashboard in real-time for glass-box transparency.
    """
    session_id: str
    current_stage: StageEnum = StageEnum.INPUT
    status: StatusEnum = StatusEnum.PENDING
    progress_percent: float = 0.0
    
    # Input metadata
    input_type: Optional[str] = None  # text, txt, pdf, docx
    input_length: Optional[int] = None
    detected_language: Optional[str] = None
    
    # Stage tracking
    stages_completed: List[StageEnum] = Field(default_factory=list)
    stage_metadata: Dict[str, StageMetadata] = Field(default_factory=dict)
    
    # Task and tool info
    detected_task: Optional[str] = None
    selected_tool: Optional[str] = None
    task_confidence: Optional[float] = None
    
    # Intervention flags
    is_paused: bool = False
    has_override: bool = False
    override_details: Optional[Dict[str, Any]] = None
    
    # Logs for transparency
    logs: List[LogEntry] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Draft output
    draft_output: Optional[Dict[str, Any]] = None
    draft_approved: bool = False
    
    # Final output
    final_output: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_stage: Optional[StageEnum] = None
    
    def add_log(self, message: str, level: LogLevel = LogLevel.INFO, 
                stage: Optional[StageEnum] = None, metadata: Optional[Dict] = None):
        """Add a log entry to the stream state."""
        entry = LogEntry(
            level=level,
            stage=stage or self.current_stage,
            message=message,
            metadata=metadata
        )
        self.logs.append(entry)
        self.updated_at = datetime.utcnow()
        return entry
    
    def update_stage(self, stage: StageEnum, status: StatusEnum = StatusEnum.RUNNING):
        """Update the current pipeline stage."""
        self.current_stage = stage
        self.status = status
        self.updated_at = datetime.utcnow()
        
        # Initialize stage metadata if not exists
        if stage.value not in self.stage_metadata:
            self.stage_metadata[stage.value] = StageMetadata(
                stage=stage,
                started_at=datetime.utcnow()
            )
    
    def complete_stage(self, stage: StageEnum, output_summary: Optional[str] = None):
        """Mark a stage as completed."""
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)
        
        if stage.value in self.stage_metadata:
            meta = self.stage_metadata[stage.value]
            meta.completed_at = datetime.utcnow()
            if meta.started_at:
                meta.duration_ms = (meta.completed_at - meta.started_at).total_seconds() * 1000
            meta.output_summary = output_summary
        
        # Calculate progress
        total_stages = len(StageEnum)
        self.progress_percent = (len(self.stages_completed) / total_stages) * 100
        self.updated_at = datetime.utcnow()
    
    def to_broadcast(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for WebSocket broadcast."""
        return self.model_dump(mode='json')
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
