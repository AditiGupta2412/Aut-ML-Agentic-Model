"""
AutoBench Configuration
Global settings and feature flags for the Glass-Box AI system.
"""
from pathlib import Path
from typing import Optional
import os

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application Info
    APP_NAME: str = "AutoBench"
    APP_VERSION: str = "1.0.0-MVP"
    DEBUG: bool = True
    
    # Server Settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    
    # Model Paths
    BASE_DIR: Path = Path(__file__).parent
    MODELS_DIR: Path = BASE_DIR / "models_cache"
    
    # NLP Model Settings
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"
    NER_MODEL: str = "en_core_web_sm"
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
    
    # Pipeline Settings
    MAX_TEXT_LENGTH: int = 50000
    DEFAULT_LANGUAGE: str = "en"
    
    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # Execution Engine Settings
    ENABLE_PAUSE_RESUME: bool = True
    ENABLE_TOOL_OVERRIDE: bool = True
    ENABLE_TASK_OVERRIDE: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DECISIONS: bool = True
    LOG_INTERMEDIATE_OUTPUTS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure models directory exists
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Task type definitions
class TaskType:
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TEXT_CLASSIFICATION = "text_classification"
    NER = "named_entity_recognition"
    TOPIC_MODELING = "topic_modeling"
    GENERAL_ANALYSIS = "general_analysis"
    
    ALL_TASKS = [
        SENTIMENT_ANALYSIS,
        TEXT_CLASSIFICATION,
        NER,
        TOPIC_MODELING,
        GENERAL_ANALYSIS
    ]


# Pipeline stage definitions
class PipelineStage:
    INPUT = "input"
    PREPROCESSING = "preprocessing"
    TASK_DETECTION = "task_detection"
    TOOL_SELECTION = "tool_selection"
    EXECUTION = "execution"
    AGGREGATION = "aggregation"
    DRAFT_OUTPUT = "draft_output"
    FINAL_OUTPUT = "final_output"
    
    ALL_STAGES = [
        INPUT,
        PREPROCESSING,
        TASK_DETECTION,
        TOOL_SELECTION,
        EXECUTION,
        AGGREGATION,
        DRAFT_OUTPUT,
        FINAL_OUTPUT
    ]


# Pipeline status definitions
class PipelineStatus:
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"
    WAITING_APPROVAL = "waiting_approval"
