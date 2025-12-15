"""
WebSocket Routes
Real-time streaming endpoints for dashboard.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional

try:
    from ..services.websocket_manager import ws_manager
    from ..services.state_manager import state_manager
except ImportError:
    from services.websocket_manager import ws_manager
    from services.state_manager import state_manager


router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/{session_id}")
async def websocket_session_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for session-specific updates."""
    await ws_manager.connect(websocket, session_id)
    
    try:
        state = state_manager.get_session(session_id)
        if state:
            await websocket.send_json({"type": "initial_state", "state": state.to_broadcast()})
        
        while True:
            try:
                data = await websocket.receive_json()
                await handle_websocket_message(websocket, session_id, data)
            except Exception:
                pass
                
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, session_id)


@router.websocket("/ws")
async def websocket_global_endpoint(websocket: WebSocket):
    """Global WebSocket endpoint for monitoring all sessions."""
    await ws_manager.connect(websocket, session_id=None)
    
    try:
        sessions = state_manager.get_all_sessions()
        await websocket.send_json({
            "type": "active_sessions",
            "count": len(sessions),
            "sessions": [
                {"session_id": s.session_id, "status": s.status.value, "stage": s.current_stage.value}
                for s in sessions.values()
            ]
        })
        
        while True:
            try:
                data = await websocket.receive_json()
                await handle_global_message(websocket, data)
            except Exception:
                pass
                
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, session_id=None)


async def handle_websocket_message(websocket: WebSocket, session_id: str, data: dict):
    """Handle incoming WebSocket messages from clients."""
    message_type = data.get("type")
    
    if message_type == "ping":
        await websocket.send_json({"type": "pong"})
    elif message_type == "get_state":
        state = state_manager.get_session(session_id)
        if state:
            await websocket.send_json({"type": "state", "state": state.to_broadcast()})
    elif message_type == "pause":
        success = await state_manager.pause(session_id)
        await websocket.send_json({"type": "pause_response", "success": success})
    elif message_type == "resume":
        success = await state_manager.resume(session_id)
        await websocket.send_json({"type": "resume_response", "success": success})
    elif message_type == "abort":
        reason = data.get("reason", "User requested via WebSocket")
        success = await state_manager.abort(session_id, reason)
        await websocket.send_json({"type": "abort_response", "success": success})


async def handle_global_message(websocket: WebSocket, data: dict):
    """Handle incoming messages on global WebSocket."""
    message_type = data.get("type")
    
    if message_type == "ping":
        await websocket.send_json({"type": "pong"})
    elif message_type == "list_sessions":
        sessions = state_manager.get_all_sessions()
        await websocket.send_json({
            "type": "sessions_list",
            "count": len(sessions),
            "sessions": [
                {"session_id": s.session_id, "status": s.status.value, "stage": s.current_stage.value, "progress": s.progress_percent}
                for s in sessions.values()
            ]
        })
