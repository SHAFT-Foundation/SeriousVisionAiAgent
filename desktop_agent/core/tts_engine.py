"""
Text-to-speech engine for Vision Agent
"""
import logging
import threading
import queue
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pyttsx3
import time

logger = logging.getLogger(__name__)


class TTSState(Enum):
    """TTS engine states"""
    IDLE = "idle"
    SPEAKING = "speaking"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class TTSSettings:
    """Text-to-speech settings"""
    rate: int = 200  # Words per minute
    volume: float = 0.9  # 0.0 to 1.0
    voice_id: Optional[str] = None
    pitch: Optional[float] = None  # Platform specific
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rate": self.rate,
            "volume": self.volume,
            "voice_id": self.voice_id,
            "pitch": self.pitch
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TTSSettings':
        return cls(
            rate=data.get("rate", 200),
            volume=data.get("volume", 0.9),
            voice_id=data.get("voice_id"),
            pitch=data.get("pitch")
        )


@dataclass
class TTSMessage:
    """Message to be spoken"""
    text: str
    priority: int = 1  # 1=low, 2=medium, 3=high
    interrupt: bool = False
    callback: Optional[Callable] = None
    message_id: Optional[str] = None


class TTSEngine:
    """Cross-platform text-to-speech engine"""
    
    def __init__(self, settings: TTSSettings = None):
        """Initialize TTS engine"""
        self.settings = settings or TTSSettings()
        self.state = TTSState.IDLE
        
        # Initialize pyttsx3 engine
        self.engine = None
        self._init_engine()
        
        # Message queue for managing speech
        self.message_queue = queue.PriorityQueue()
        self.current_message: Optional[TTSMessage] = None
        
        # Threading
        self.worker_thread = None
        self.should_stop = threading.Event()
        
        # Callbacks
        self.on_start_callback: Optional[Callable] = None
        self.on_finish_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        
        logger.info("TTS engine initialized")
    
    def _init_engine(self):
        """Initialize the pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Set initial properties
            self.engine.setProperty('rate', self.settings.rate)
            self.engine.setProperty('volume', self.settings.volume)
            
            # Set voice if specified
            if self.settings.voice_id:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if voice.id == self.settings.voice_id:
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Setup callbacks
            self.engine.connect('started-utterance', self._on_utterance_start)
            self.engine.connect('finished-utterance', self._on_utterance_finish)
            
            logger.info("pyttsx3 engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def start(self):
        """Start the TTS worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.should_stop.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("TTS worker thread started")
    
    def stop(self):
        """Stop the TTS engine"""
        self.should_stop.set()
        self.stop_speaking()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        
        logger.info("TTS engine stopped")
    
    def speak(self, text: str, priority: int = 1, interrupt: bool = False,
             callback: Optional[Callable] = None, message_id: str = None):
        """
        Add text to speech queue
        
        Args:
            text: Text to speak
            priority: Message priority (1=low, 2=medium, 3=high)
            interrupt: Whether to interrupt current speech
            callback: Function to call when speech finishes
            message_id: Unique identifier for this message
        """
        if not text.strip():
            return
        
        message = TTSMessage(
            text=text,
            priority=priority,
            interrupt=interrupt,
            callback=callback,
            message_id=message_id
        )
        
        if interrupt and self.state == TTSState.SPEAKING:
            self.stop_speaking()
            self.clear_queue()
        
        # Add to priority queue (lower number = higher priority)
        self.message_queue.put((4 - priority, time.time(), message))
        
        # Start worker if not running
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.start()
    
    def speak_immediately(self, text: str, callback: Optional[Callable] = None):
        """Speak text immediately, interrupting current speech"""
        self.speak(text, priority=3, interrupt=True, callback=callback)
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.engine and self.state == TTSState.SPEAKING:
            try:
                self.engine.stop()
                self.state = TTSState.STOPPED
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
    
    def pause_speaking(self):
        """Pause current speech (if supported)"""
        # pyttsx3 doesn't support pause/resume, so we'll stop instead
        self.stop_speaking()
        self.state = TTSState.PAUSED
    
    def resume_speaking(self):
        """Resume paused speech"""
        if self.state == TTSState.PAUSED:
            self.state = TTSState.IDLE
    
    def clear_queue(self):
        """Clear the message queue"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.state == TTSState.SPEAKING
    
    def get_queue_size(self) -> int:
        """Get number of messages in queue"""
        return self.message_queue.qsize()
    
    def update_settings(self, settings: TTSSettings):
        """Update TTS settings"""
        self.settings = settings
        
        if self.engine:
            try:
                self.engine.setProperty('rate', settings.rate)
                self.engine.setProperty('volume', settings.volume)
                
                if settings.voice_id:
                    self.engine.setProperty('voice', settings.voice_id)
                
                logger.info("TTS settings updated")
            except Exception as e:
                logger.error(f"Error updating TTS settings: {e}")
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if not self.engine:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_info = {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": getattr(voice, 'languages', []),
                    "gender": getattr(voice, 'gender', 'unknown'),
                    "age": getattr(voice, 'age', 'unknown')
                }
                voice_list.append(voice_info)
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def _worker_loop(self):
        """Main worker loop for processing speech queue"""
        while not self.should_stop.is_set():
            try:
                # Get next message from queue (timeout to check stop condition)
                try:
                    priority, timestamp, message = self.message_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Speak the message
                self._speak_message(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in TTS worker loop: {e}")
                time.sleep(0.1)
    
    def _speak_message(self, message: TTSMessage):
        """Speak a single message"""
        if self.should_stop.is_set() or not self.engine:
            return
        
        try:
            self.current_message = message
            self.state = TTSState.SPEAKING
            
            # Call start callback
            if self.on_start_callback:
                try:
                    self.on_start_callback(message.text)
                except Exception as e:
                    logger.error(f"Error in start callback: {e}")
            
            # Speak the text
            self.engine.say(message.text)
            self.engine.runAndWait()
            
            # Call message-specific callback
            if message.callback:
                try:
                    message.callback(message.text, True)  # True = completed
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
            
        except Exception as e:
            logger.error(f"Error speaking message: {e}")
            
            # Call error callback
            if self.on_error_callback:
                try:
                    self.on_error_callback(str(e))
                except Exception as e:
                    logger.error(f"Error in error callback: {e}")
            
            # Call message callback with error
            if message.callback:
                try:
                    message.callback(message.text, False)  # False = error
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
        
        finally:
            self.current_message = None
            if self.state == TTSState.SPEAKING:
                self.state = TTSState.IDLE
    
    def _on_utterance_start(self, name):
        """Called when utterance starts"""
        logger.debug(f"TTS utterance started: {name}")
    
    def _on_utterance_finish(self, name, completed):
        """Called when utterance finishes"""
        logger.debug(f"TTS utterance finished: {name}, completed: {completed}")
        
        if self.on_finish_callback:
            try:
                self.on_finish_callback(completed)
            except Exception as e:
                logger.error(f"Error in finish callback: {e}")


class AccessibilityTTS:
    """High-level TTS interface for accessibility"""
    
    def __init__(self, settings: TTSSettings = None):
        self.engine = TTSEngine(settings)
        self.engine.start()
        
        # Predefined message templates
        self.templates = {
            "processing": "Processing screen content...",
            "ready": "Screen analysis ready",
            "error": "An error occurred during processing",
            "no_changes": "No screen changes detected",
            "cache_hit": "Using cached results"
        }
    
    def announce_result(self, alt_text: str, interrupt: bool = False):
        """Announce accessibility results"""
        self.engine.speak(alt_text, priority=2, interrupt=interrupt)
    
    def announce_status(self, status: str, interrupt: bool = False):
        """Announce system status"""
        message = self.templates.get(status, status)
        self.engine.speak(message, priority=1, interrupt=interrupt)
    
    def announce_urgent(self, text: str):
        """Announce urgent message"""
        self.engine.speak_immediately(text)
    
    def set_voice_settings(self, rate: int = None, volume: float = None, 
                          voice_id: str = None):
        """Update voice settings"""
        current = self.engine.settings
        new_settings = TTSSettings(
            rate=rate if rate is not None else current.rate,
            volume=volume if volume is not None else current.volume,
            voice_id=voice_id if voice_id is not None else current.voice_id,
            pitch=current.pitch
        )
        self.engine.update_settings(new_settings)
    
    def get_voices(self) -> list:
        """Get available voices"""
        return self.engine.get_available_voices()
    
    def stop(self):
        """Stop TTS engine"""
        self.engine.stop()


# Global TTS instance for easy access
_global_tts: Optional[AccessibilityTTS] = None

def get_tts() -> AccessibilityTTS:
    """Get global TTS instance"""
    global _global_tts
    if _global_tts is None:
        _global_tts = AccessibilityTTS()
    return _global_tts

def speak(text: str, interrupt: bool = False):
    """Quick speak function"""
    get_tts().announce_result(text, interrupt=interrupt)