"""
Tool Execution Schema
Data models for tracking NLP tool/model execution.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ExecutionStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class IntermediateOutput(BaseModel):
    """Intermediate output from tool execution for explainability."""
    step_name: str
    step_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]
    description: str


class ToolExecution(BaseModel):
    """
    Complete execution record for an NLP tool/model.
    Provides full traceability for glass-box transparency.
    """
    session_id: str
    execution_id: str
    
    # Tool identification
    tool_name: str
    tool_type: str  # sentiment, ner, topic, classification
    model_name: str
    model_version: str
    
    # Configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution state
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress_percent: float = 0.0
    current_step: Optional[str] = None
    
    # Timing
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Intermediate outputs for explainability
    intermediate_outputs: List[IntermediateOutput] = Field(default_factory=list)
    
    # Final output
    final_output: Optional[Dict[str, Any]] = None
    output_confidence: Optional[float] = None
    
    # Override tracking
    was_overridden: bool = False
    replacement_tool: Optional[str] = None
    override_reason: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Resource usage (optional)
    memory_usage_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    
    def start(self):
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self, output: Dict[str, Any], confidence: Optional[float] = None):
        """Mark execution as completed with output."""
        self.status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.final_output = output
        self.output_confidence = confidence
        if self.started_at:
            self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def fail(self, error_message: str, error_details: Optional[Dict] = None):
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_details = error_details
    
    def add_intermediate(self, step_name: str, data: Dict[str, Any], description: str):
        """Add an intermediate output for explainability."""
        step_num = len(self.intermediate_outputs) + 1
        output = IntermediateOutput(
            step_name=step_name,
            step_number=step_num,
            data=data,
            description=description
        )
        self.intermediate_outputs.append(output)
        self.current_step = step_name
        return output
    
    def to_summary(self) -> Dict[str, Any]:
        """Generate execution summary for logging/display."""
        return {
            "tool": self.tool_name,
            "model": self.model_name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "confidence": self.output_confidence,
            "steps_completed": len(self.intermediate_outputs),
            "was_overridden": self.was_overridden
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
