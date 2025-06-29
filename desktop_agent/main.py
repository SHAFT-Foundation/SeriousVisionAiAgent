"""
Main entry point for Vision Agent desktop application
"""
import asyncio
import logging
import sys
import os
import signal
from typing import Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop_agent.core.screen_monitor import ScreenMonitor, capture_primary_screen
from desktop_agent.core.image_processor import ImageProcessor, ProcessingConfig
from desktop_agent.core.tts_engine import AccessibilityTTS, TTSSettings
from desktop_agent.core.hotkey_manager import HotkeyManager, HotkeyAction
from desktop_agent.core.api_client import VisionAgentClient, ServerConfig, ProcessingRequest

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration"""
    server_url: str = "http://localhost:8000"
    user_id: str = "default_user"
    default_context: str = "general"
    default_verbosity: str = "medium"
    monitor_id: int = 1
    auto_monitoring: bool = False
    tts_enabled: bool = True
    debug: bool = False


class VisionAgentApp:
    """Main Vision Agent desktop application"""
    
    def __init__(self, config: AppConfig = None):
        """Initialize Vision Agent application"""
        self.config = config or AppConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Core components
        self.screen_monitor: Optional[ScreenMonitor] = None
        self.image_processor: Optional[ImageProcessor] = None
        self.tts: Optional[AccessibilityTTS] = None
        self.hotkey_manager: Optional[HotkeyManager] = None
        self.api_client: Optional[VisionAgentClient] = None
        
        # State
        self.is_running = False
        self.monitoring_enabled = False
        self.last_result: Optional[str] = None
        self.current_verbosity = self.config.default_verbosity
        
        logger.info("Vision Agent application initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.config.debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("vision_agent.log")
            ]
        )
    
    async def start(self):
        """Start the Vision Agent application"""
        logger.info("Starting Vision Agent...")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Setup hotkey callbacks
            self._setup_hotkey_callbacks()
            
            # Start hotkey listener
            self.hotkey_manager.start_listening()
            
            # Test server connection
            if not await self.api_client.test_connection():
                logger.warning("Cannot connect to server. Some features may not work.")
                if self.tts:
                    self.tts.announce_status("error")
            else:
                logger.info("Connected to Vision Agent server")
                if self.tts:
                    self.tts.announce_status("ready")
            
            self.is_running = True
            
            # Announce startup
            if self.tts:
                help_text = self.hotkey_manager.get_help_text()
                self.tts.announce_result(
                    f"Vision Agent started. {help_text[:200]}...", 
                    interrupt=False
                )
            
            logger.info("Vision Agent started successfully")
            
            # Start monitoring if enabled
            if self.config.auto_monitoring:
                await self._toggle_monitoring()
            
        except Exception as e:
            logger.error(f"Failed to start Vision Agent: {e}")
            raise
    
    async def stop(self):
        """Stop the Vision Agent application"""
        logger.info("Stopping Vision Agent...")
        
        self.is_running = False
        self.monitoring_enabled = False
        
        # Stop components
        if self.hotkey_manager:
            self.hotkey_manager.stop_listening()
        
        if self.tts:
            self.tts.stop()
        
        if self.api_client:
            await self.api_client.close()
        
        logger.info("Vision Agent stopped")
    
    async def _initialize_components(self):
        """Initialize all application components"""
        
        # Initialize screen monitor
        self.screen_monitor = ScreenMonitor(
            change_threshold=0.05,
            max_fps=2  # Limit capture rate for performance
        )
        
        # Initialize image processor
        processing_config = ProcessingConfig(
            max_size=2048,
            jpeg_quality=85,
            enhance_contrast=1.2,
            auto_crop_whitespace=True
        )
        self.image_processor = ImageProcessor(processing_config)
        
        # Initialize TTS
        if self.config.tts_enabled:
            tts_settings = TTSSettings(
                rate=200,
                volume=0.9
            )
            self.tts = AccessibilityTTS(tts_settings)
        
        # Initialize hotkey manager
        self.hotkey_manager = HotkeyManager()
        
        # Initialize API client
        server_config = ServerConfig(base_url=self.config.server_url)
        self.api_client = VisionAgentClient(server_config)
        
        logger.info("All components initialized")
    
    def _setup_hotkey_callbacks(self):
        """Setup callbacks for hotkey actions"""
        
        # Register action callbacks
        callbacks = {
            HotkeyAction.CAPTURE_SCREEN: self._capture_screen,
            HotkeyAction.REPEAT_LAST: self._repeat_last_result,
            HotkeyAction.TOGGLE_MONITORING: self._toggle_monitoring,
            HotkeyAction.STOP_SPEECH: self._stop_speech,
            HotkeyAction.PAUSE_SPEECH: self._pause_speech,
            HotkeyAction.INCREASE_VERBOSITY: self._increase_verbosity,
            HotkeyAction.DECREASE_VERBOSITY: self._decrease_verbosity,
            HotkeyAction.TOGGLE_TTS: self._toggle_tts,
            HotkeyAction.SHOW_HELP: self._show_help,
            HotkeyAction.QUIT_APP: self._quit_app
        }
        
        for action, callback in callbacks.items():
            self.hotkey_manager.register_action_callback(action, callback)
    
    async def _capture_screen(self):
        """Capture and process current screen"""
        try:
            logger.info("Capturing screen...")
            
            if self.tts:
                self.tts.announce_status("processing")
            
            # Capture screen
            capture_result = capture_primary_screen()
            
            # Process image
            processed = self.image_processor.process_image(
                capture_result.image_data,
                content_type=self.config.default_context
            )
            
            # Send to server for analysis
            request = ProcessingRequest(
                image_data=processed.image_data,
                user_id=self.config.user_id,
                context=self.config.default_context,
                verbosity=self.current_verbosity
            )
            
            response = await self.api_client.process_image(request)
            
            if response.success:
                # Get appropriate description based on verbosity
                description = self._get_description_for_verbosity(response)
                
                # Store last result
                self.last_result = description
                
                # Announce result
                if self.tts:
                    self.tts.announce_result(description, interrupt=True)
                
                logger.info(f"Screen processed successfully (confidence: {response.confidence_score:.2f})")
                
            else:
                error_msg = f"Processing failed: {response.error_message}"
                logger.error(error_msg)
                
                if self.tts:
                    self.tts.announce_urgent("Processing failed")
                    
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            if self.tts:
                self.tts.announce_urgent("Capture error occurred")
    
    def _get_description_for_verbosity(self, response) -> str:
        """Get description based on current verbosity level"""
        if self.current_verbosity == "brief":
            return response.alt_text or "No description available"
        elif self.current_verbosity == "detailed":
            return response.detailed_description or response.alt_text or "No description available"
        else:  # medium
            return response.alt_text or "No description available"
    
    async def _repeat_last_result(self):
        """Repeat the last processing result"""
        if self.last_result and self.tts:
            self.tts.announce_result(self.last_result, interrupt=True)
        elif self.tts:
            self.tts.announce_urgent("No previous result to repeat")
    
    async def _toggle_monitoring(self):
        """Toggle automatic screen monitoring"""
        self.monitoring_enabled = not self.monitoring_enabled
        
        status = "enabled" if self.monitoring_enabled else "disabled"
        logger.info(f"Screen monitoring {status}")
        
        if self.tts:
            self.tts.announce_urgent(f"Monitoring {status}")
        
        if self.monitoring_enabled:
            # Start monitoring in background
            asyncio.create_task(self._monitor_screen_changes())
    
    async def _monitor_screen_changes(self):
        """Monitor screen for changes and process automatically"""
        logger.info("Starting automatic screen monitoring")
        
        while self.monitoring_enabled and self.is_running:
            try:
                if self.screen_monitor.has_screen_changed(self.config.monitor_id):
                    await self._capture_screen()
                
                # Wait before next check
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error in screen monitoring: {e}")
                await asyncio.sleep(5.0)  # Longer wait on error
        
        logger.info("Screen monitoring stopped")
    
    def _stop_speech(self):
        """Stop current speech"""
        if self.tts:
            self.tts.stop()
    
    def _pause_speech(self):
        """Pause/resume speech"""
        # TTS engine doesn't support pause, so we'll stop instead
        if self.tts:
            self.tts.stop()
    
    def _increase_verbosity(self):
        """Increase description verbosity"""
        levels = ["brief", "medium", "detailed"]
        current_index = levels.index(self.current_verbosity)
        
        if current_index < len(levels) - 1:
            self.current_verbosity = levels[current_index + 1]
            
            if self.tts:
                self.tts.announce_urgent(f"Verbosity: {self.current_verbosity}")
    
    def _decrease_verbosity(self):
        """Decrease description verbosity"""
        levels = ["brief", "medium", "detailed"]
        current_index = levels.index(self.current_verbosity)
        
        if current_index > 0:
            self.current_verbosity = levels[current_index - 1]
            
            if self.tts:
                self.tts.announce_urgent(f"Verbosity: {self.current_verbosity}")
    
    def _toggle_tts(self):
        """Toggle text-to-speech on/off"""
        if self.tts:
            # Simple toggle by recreating TTS or not
            # In a real implementation, you'd have a proper enable/disable
            self.tts.announce_urgent("TTS toggle not fully implemented")
    
    def _show_help(self):
        """Show help and hotkey information"""
        if self.tts:
            help_text = self.hotkey_manager.get_help_text()
            self.tts.announce_result(help_text, interrupt=True)
    
    def _quit_app(self):
        """Quit the application"""
        logger.info("Quit requested via hotkey")
        
        if self.tts:
            self.tts.announce_urgent("Goodbye")
        
        # Use asyncio to stop the app
        asyncio.create_task(self._shutdown())
    
    async def _shutdown(self):
        """Graceful shutdown"""
        # Give TTS time to speak goodbye
        await asyncio.sleep(1.0)
        await self.stop()
        
        # Exit the application
        os._exit(0)
    
    async def run(self):
        """Main application run loop"""
        await self.start()
        
        try:
            # Keep the application running
            while self.is_running:
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.stop()


def setup_signal_handlers(app: VisionAgentApp):
    """Setup signal handlers for graceful shutdown"""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(app.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    # Parse command line arguments (basic version)
    config = AppConfig()
    
    if len(sys.argv) > 1:
        if "--debug" in sys.argv:
            config.debug = True
        if "--no-tts" in sys.argv:
            config.tts_enabled = False
        if "--auto-monitor" in sys.argv:
            config.auto_monitoring = True
    
    # Create and run application
    app = VisionAgentApp(config)
    setup_signal_handlers(app)
    
    try:
        await app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we're on Windows and need to set the event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())