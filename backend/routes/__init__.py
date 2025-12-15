"""
AutoBench Routes
API route modules.
"""
try:
    from .pipeline import router as pipeline_router
    from .websocket import router as websocket_router
except ImportError:
    from pipeline import router as pipeline_router
    from websocket import router as websocket_router

__all__ = ["pipeline_router", "websocket_router"]
