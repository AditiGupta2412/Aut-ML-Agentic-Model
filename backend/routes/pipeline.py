"""
Pipeline API Routes
REST endpoints for pipeline control and interaction.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio

try:
    from ..services.state_manager import state_manager
    from ..services.websocket_manager import ws_manager
    from ..services.pipeline_orchestrator import PipelineOrchestrator
    from ..services.model_config import model_config_manager
    from ..modules.export_utils import export_manager
except ImportError:
    from services.state_manager import state_manager
    from services.websocket_manager import ws_manager
    from services.pipeline_orchestrator import PipelineOrchestrator
    from services.model_config import model_config_manager
    from modules.export_utils import export_manager


router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

# Initialize orchestrator
orchestrator = PipelineOrchestrator(state_manager, ws_manager)


# Request/Response Models
class StartPipelineRequest(BaseModel):
    text: Optional[str] = None
    force_task: Optional[str] = None
    force_tool: Optional[str] = None


class OverrideTaskRequest(BaseModel):
    task: str
    reason: Optional[str] = "User override"


class OverrideToolRequest(BaseModel):
    tool: str
    reason: Optional[str] = "User override"


class ApproveRequest(BaseModel):
    notes: Optional[str] = None


class EditDraftRequest(BaseModel):
    summary: Optional[str] = None
    content: Optional[dict] = None


class SettingsUpdateRequest(BaseModel):
    sentiment_threshold: Optional[float] = None
    ner_entity_types: Optional[List[str]] = None
    topic_count: Optional[int] = None
    classification_threshold: Optional[float] = None
    classification_categories: Optional[List[str]] = None
    max_text_length: Optional[int] = None
    auto_detect_task: Optional[bool] = None
    preferred_task: Optional[str] = None
    export_include_raw: Optional[bool] = None


# Pipeline Control Endpoints
@router.post("/start")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(None),
    force_task: Optional[str] = Form(None),
    force_tool: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Start a new analysis pipeline."""
    if not text and not file:
        raise HTTPException(status_code=400, detail="Provide either text or a file")
    
    input_type = "text" if text else file.filename.split(".")[-1] if file else "unknown"
    state = state_manager.create_session(input_type)
    session_id = state.session_id
    
    file_content = None
    filename = None
    if file:
        file_content = await file.read()
        filename = file.filename
    
    # Start pipeline as a background task - use asyncio.create_task directly
    asyncio.create_task(
        orchestrator.run_pipeline(
            session_id=session_id,
            text=text,
            file_content=file_content,
            filename=filename,
            force_task=force_task,
            force_tool=force_tool
        )
    )
    
    return {
        "session_id": session_id,
        "status": "started",
        "websocket_url": f"/ws/{session_id}",
        "message": "Pipeline started. Connect to WebSocket for real-time updates."
    }


@router.post("/start/json")
async def start_pipeline_json(request: StartPipelineRequest, background_tasks: BackgroundTasks):
    """Start a new analysis pipeline with JSON request body."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    state = state_manager.create_session("text")
    session_id = state.session_id
    
    asyncio.create_task(
        orchestrator.run_pipeline(
            session_id=session_id,
            text=request.text,
            force_task=request.force_task,
            force_tool=request.force_tool
        )
    )
    
    return {"session_id": session_id, "status": "started", "websocket_url": f"/ws/{session_id}"}


@router.get("/{session_id}/state")
async def get_pipeline_state(session_id: str):
    """Get the current state of a pipeline session."""
    state = state_manager.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "state": state.to_broadcast()}


@router.post("/{session_id}/pause")
async def pause_pipeline(session_id: str):
    """Pause a running pipeline."""
    success = await state_manager.pause(session_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot pause session")
    return {"session_id": session_id, "status": "paused"}


@router.post("/{session_id}/resume")
async def resume_pipeline(session_id: str):
    """Resume a paused pipeline."""
    success = await state_manager.resume(session_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot resume session")
    return {"session_id": session_id, "status": "resumed"}


@router.post("/{session_id}/abort")
async def abort_pipeline(session_id: str):
    """Abort a running pipeline."""
    success = await state_manager.abort(session_id, "User requested abort")
    if not success:
        raise HTTPException(status_code=400, detail="Cannot abort session")
    return {"session_id": session_id, "status": "aborted"}


@router.post("/{session_id}/override/task")
async def override_task(session_id: str, request: OverrideTaskRequest):
    """Override the detected task for a session."""
    success = await state_manager.override_task(session_id, request.task, request.reason)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot override task")
    return {"session_id": session_id, "new_task": request.task, "status": "task_overridden"}


@router.post("/{session_id}/override/tool")
async def override_tool(session_id: str, request: OverrideToolRequest):
    """Override the selected tool for a session."""
    success = await state_manager.override_tool(session_id, request.tool, request.reason)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot override tool")
    return {"session_id": session_id, "new_tool": request.tool, "status": "tool_overridden"}


@router.get("/{session_id}/draft")
async def get_draft(session_id: str):
    """Get the draft output for review."""
    draft = orchestrator.output_generator.get_draft(session_id)
    if not draft:
        raise HTTPException(status_code=404, detail="No draft available")
    return {"session_id": session_id, "draft": draft}


@router.post("/{session_id}/draft/approve")
async def approve_draft(session_id: str, request: ApproveRequest = None):
    """Approve the draft and generate final output."""
    notes = request.notes if request else None
    try:
        result = await orchestrator.approve_draft(session_id, notes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{session_id}/final")
async def get_final_output(session_id: str):
    """Get the final approved output."""
    final = orchestrator.output_generator.get_final(session_id)
    if not final:
        raise HTTPException(status_code=404, detail="No final output available")
    return {"session_id": session_id, "final_output": final}


@router.get("/tasks/available")
async def get_available_tasks():
    """Get list of available NLP tasks."""
    return {"tasks": orchestrator.task_detector.get_available_tasks()}


@router.get("/tools/available")
async def get_available_tools(task: Optional[str] = None):
    """Get list of available NLP tools."""
    return {"tools": orchestrator.tool_selector.get_available_tools(task)}


@router.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = state_manager.get_all_sessions()
    return {
        "count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "stage": s.current_stage.value,
                "progress": s.progress_percent,
                "created_at": s.created_at.isoformat()
            }
            for s in sessions.values()
        ]
    }


# ============== Export Endpoints ==============

@router.get("/{session_id}/export/json")
async def export_json(session_id: str, include_raw: bool = False):
    """Export analysis result as JSON."""
    state = state_manager.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get result (draft or final)
    result = state.final_output or state.draft_output
    if not result:
        raise HTTPException(status_code=400, detail="No result available to export")
    
    export_data = export_manager.export_json(result, include_raw=include_raw)
    
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename=autobench_{session_id[:8]}.json"
        }
    )


@router.get("/{session_id}/export/pdf")
async def export_pdf(session_id: str):
    """Export analysis result as PDF report."""
    state = state_manager.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get result (draft or final)
    result = state.final_output or state.draft_output
    if not result:
        raise HTTPException(status_code=400, detail="No result available to export")
    
    pdf_bytes = export_manager.export_pdf(result, state.to_broadcast())
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=autobench_report_{session_id[:8]}.pdf"
        }
    )


# ============== Settings Endpoints ==============

@router.get("/settings")
async def get_settings(session_id: Optional[str] = None):
    """Get current settings."""
    settings = model_config_manager.get_settings(session_id)
    return {"settings": settings.to_dict()}


@router.post("/settings")
async def update_settings(request: SettingsUpdateRequest, session_id: Optional[str] = None):
    """Update settings."""
    if not session_id:
        session_id = "global"
    
    # Build update dict from non-None fields
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    
    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update")
    
    new_settings = model_config_manager.update_settings(session_id, updates)
    return {"message": "Settings updated", "settings": new_settings.to_dict()}


@router.post("/settings/reset")
async def reset_settings(session_id: Optional[str] = None):
    """Reset settings to defaults."""
    if not session_id:
        session_id = "global"
    
    settings = model_config_manager.reset_settings(session_id)
    return {"message": "Settings reset to defaults", "settings": settings.to_dict()}


@router.get("/models/config")
async def get_model_configs():
    """Get all model configurations."""
    return {"models": model_config_manager.get_all_configs()}


@router.post("/models/{tool_id}/config")
async def update_model_config(tool_id: str, updates: Dict[str, Any]):
    """Update configuration for a specific model."""
    config = model_config_manager.update_model_config(tool_id, updates)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model not found: {tool_id}")
    return {"message": f"Model {tool_id} configuration updated", "config": config.to_dict()}
