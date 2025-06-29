"""
Core functionality for Vision Agent desktop client
"""
from .screen_monitor import ScreenMonitor, capture_primary_screen, get_screen_info
from .image_processor import ImageProcessor, ProcessingConfig, optimize_for_llm
from .tts_engine import AccessibilityTTS, TTSSettings, get_tts, speak
from .hotkey_manager import HotkeyManager, HotkeyAction, get_hotkey_manager
from .api_client import VisionAgentClient, ServerConfig, ProcessingRequest, ProcessingResponse

__all__ = [
    "ScreenMonitor",
    "capture_primary_screen", 
    "get_screen_info",
    "ImageProcessor",
    "ProcessingConfig",
    "optimize_for_llm",
    "AccessibilityTTS",
    "TTSSettings",
    "get_tts",
    "speak",
    "HotkeyManager",
    "HotkeyAction",
    "get_hotkey_manager",
    "VisionAgentClient",
    "ServerConfig", 
    "ProcessingRequest",
    "ProcessingResponse"
]