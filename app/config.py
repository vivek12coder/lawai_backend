import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Legal AI System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Storage Configuration
    DATA_DIR: str = "data"
    QA_FILE: str = "qa_data.json"
    
    # Model Configuration
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_RESULTS: int = 5
    
    # CORS Configuration
    CORS_ORIGINS: list = ["*"]  # In production, replace with actual frontend domain
    
    # Document Analysis Configuration
    MAX_DOCUMENT_SIZE: int = 10 * 1024 * 1024  # 10MB
    SUPPORTED_DOCUMENT_TYPES: list = [
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    
    class Config:
        case_sensitive = True

# Create global settings object
settings = Settings() 