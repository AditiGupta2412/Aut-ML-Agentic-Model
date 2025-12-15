"""
AutoBench - Glass-Box Agentic AI System for Automated Text Analytics
Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .config import settings
    from .routes import pipeline_router, websocket_router
    from .services.state_manager import state_manager
    from .services.websocket_manager import ws_manager
except ImportError:
    from config import settings
    from routes import pipeline_router, websocket_router
    from services.state_manager import state_manager
    from services.websocket_manager import ws_manager


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    # AutoBench - Glass-Box Agentic AI System
    
    A production-grade agentic NLP system that:
    - Autonomously detects required NLP tasks from raw text
    - Dynamically selects and executes appropriate NLP models/tools
    - Exposes internal reasoning, decision-making, and execution state in real time
    - Allows human-in-the-loop intervention via a live control dashboard
    - Supports interruption, override, reselection, and review at every pipeline stage
    - Produces traceable, explainable, and trustworthy final outputs
    
    ## Core Features
    
    - **Glass-Box Transparency**: Every stage is observable in real time
    - **Real-time Streaming**: WebSocket-based live updates
    - **Human Intervention**: Pause, resume, override, and abort capabilities
    - **Full Traceability**: Complete audit trail from input to output
    
    ## API Categories
    
    - **Pipeline Control**: Start, pause, resume, abort analysis
    - **Task/Tool Override**: Override automatic selections
    - **Draft Review**: Review and approve outputs before finalization
    - **WebSocket**: Real-time streaming for dashboard
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pipeline_router)
app.include_router(websocket_router)

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# Root endpoint - serve dashboard
@app.get("/")
async def root():
    """Serve the dashboard or return API info."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Glass-Box Agentic AI System for Automated Text Analytics",
        "docs": "/docs",
        "dashboard": "Coming soon - place frontend files in /frontend directory",
        "endpoints": {
            "start_pipeline": "POST /api/pipeline/start",
            "websocket": "WS /ws/{session_id}",
            "get_state": "GET /api/pipeline/{session_id}/state"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "active_sessions": len(state_manager.get_all_sessions()),
        "websocket_connections": ws_manager.get_connection_count()
    }


@app.get("/api/info")
async def api_info():
    """Get API information and available capabilities."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "capabilities": {
            "tasks": [
                "sentiment_analysis",
                "named_entity_recognition",
                "text_classification",
                "topic_modeling",
                "general_analysis"
            ],
            "input_types": ["text", "txt", "pdf", "docx"],
            "interventions": ["pause", "resume", "abort", "override_task", "override_tool"],
            "streaming": True,
            "draft_review": True
        },
        "pipeline_stages": [
            "input",
            "preprocessing",
            "task_detection",
            "tool_selection",
            "execution",
            "aggregation",
            "draft_output",
            "final_output"
        ]
    }


# Register state change callback for WebSocket broadcasting
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Register callback to broadcast state changes
    async def broadcast_state_change(session_id: str, state):
        await ws_manager.broadcast_state_update(session_id, state.to_broadcast())
    
    state_manager.register_state_change_callback(broadcast_state_change)
    
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} started!")
    print(f"📊 Dashboard: http://{settings.HOST}:{settings.PORT}/")
    print(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print(f"👋 {settings.APP_NAME} shutting down...")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return {
        "error": str(exc),
        "type": type(exc).__name__,
        "path": str(request.url)
    }
