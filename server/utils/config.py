"""
Configuration management for Vision Agent server
"""
import os
import yaml
from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic import BaseSettings, Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Universal Computer Vision Accessibility Agent"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    server_host: str = Field(default="0.0.0.0", env="SERVER_HOST")
    server_port: int = Field(default=8000, env="SERVER_PORT")
    server_workers: int = Field(default=1, env="SERVER_WORKERS")
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/vision_agent",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # Security
    master_password: str = Field(default="changeme", env="MASTER_PASSWORD")
    encryption_salt: str = Field(default="vision_agent_salt", env="ENCRYPTION_SALT")
    
    # Processing
    max_image_size: int = Field(default=2048, env="MAX_IMAGE_SIZE")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    processing_timeout: int = Field(default=30, env="PROCESSING_TIMEOUT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/vision_agent.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


def load_yaml_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")
    
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
        return {}


def get_llm_config(provider: str) -> Dict[str, Any]:
    """Get LLM provider configuration"""
    yaml_config = load_yaml_config()
    llm_providers = yaml_config.get("llm_providers", {})
    
    default_config = {
        "max_tokens": 1000,
        "timeout": 30,
        "temperature": 0.1
    }
    
    provider_config = llm_providers.get(provider, {})
    return {**default_config, **provider_config}


def get_accessibility_config() -> Dict[str, Any]:
    """Get accessibility configuration"""
    yaml_config = load_yaml_config()
    return yaml_config.get("accessibility", {
        "default_verbosity": "medium",
        "tts_enabled": True,
        "tts_rate": 200,
        "screen_reader_integration": True
    })


def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration"""
    yaml_config = load_yaml_config()
    return yaml_config.get("processing", {
        "max_image_size": 2048,
        "jpeg_quality": 85,
        "cache_results": True,
        "batch_size": 4
    })