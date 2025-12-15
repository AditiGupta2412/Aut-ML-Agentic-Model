"""
Task Decision Schema
Data models for task detection and decision tracking.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DetectedTask(BaseModel):
    """A single detected task with confidence score."""
    task_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    signals: List[str] = Field(default_factory=list)  # Evidence for detection


class TaskDecision(BaseModel):
    """
    Complete task decision record for glass-box transparency.
    Tracks what was detected, what was selected, and any overrides.
    """
    session_id: str
    
    # Detection results
    detected_tasks: List[DetectedTask] = Field(default_factory=list)
    
    # Selection decision
    selected_task: Optional[str] = None
    selection_rationale: str = ""
    selection_confidence: float = 0.0
    
    # Override tracking
    override_applied: bool = False
    original_task: Optional[str] = None
    override_task: Optional[str] = None
    override_reason: Optional[str] = None
    override_by: str = "user"  # user or system
    
    # Timing
    detection_started_at: datetime = Field(default_factory=datetime.utcnow)
    detection_completed_at: Optional[datetime] = None
    detection_duration_ms: Optional[float] = None
    
    # Additional context
    input_preview: Optional[str] = None  # First N chars of input
    detection_method: str = "hybrid"  # keyword, ml, hybrid
    
    def apply_override(self, new_task: str, reason: str = "User override"):
        """Apply a task override and record the change."""
        self.original_task = self.selected_task
        self.override_task = new_task
        self.selected_task = new_task
        self.override_applied = True
        self.override_reason = reason
    
    def get_top_task(self) -> Optional[DetectedTask]:
        """Get the highest confidence detected task."""
        if not self.detected_tasks:
            return None
        return max(self.detected_tasks, key=lambda t: t.confidence)
    
    def to_log_summary(self) -> Dict[str, Any]:
        """Generate a summary for logging/display."""
        return {
            "selected_task": self.selected_task,
            "confidence": self.selection_confidence,
            "alternatives": [
                {"task": t.task_name, "confidence": t.confidence}
                for t in self.detected_tasks[:3]
            ],
            "override_applied": self.override_applied,
            "rationale": self.selection_rationale
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
