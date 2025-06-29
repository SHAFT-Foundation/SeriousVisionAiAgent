"""
Screen capture and monitoring functionality using mss library
"""
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from io import BytesIO
import mss
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ScreenRegion:
    """Represents a screen region to capture"""
    left: int
    top: int
    width: int
    height: int
    monitor_id: int = 0
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    def to_mss_dict(self) -> Dict[str, int]:
        """Convert to mss monitor dictionary format"""
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height
        }


@dataclass
class CaptureResult:
    """Result of a screen capture operation"""
    image_data: bytes
    image_hash: str
    dimensions: Tuple[int, int]  # (width, height)
    monitor_info: Dict[str, Any]
    capture_time: float
    file_size: int


class ScreenMonitor:
    """Manages screen capture and change detection"""
    
    def __init__(self, change_threshold: float = 0.05, max_fps: int = 5):
        """
        Initialize screen monitor
        
        Args:
            change_threshold: Minimum change ratio to trigger new capture (0.0-1.0)
            max_fps: Maximum capture rate per second
        """
        self.change_threshold = change_threshold
        self.max_fps = max_fps
        self.min_capture_interval = 1.0 / max_fps
        
        self.sct = mss.mss()
        self.last_capture_time = 0.0
        self.last_image_hash = None
        self.last_image_gray = None
        
        # Get monitor information
        self.monitors = self.sct.monitors
        logger.info(f"Detected {len(self.monitors) - 1} monitors")
    
    def get_monitors(self) -> List[Dict[str, Any]]:
        """Get list of available monitors"""
        monitors_info = []
        for i, monitor in enumerate(self.monitors[1:], 1):  # Skip "all monitors" at index 0
            monitors_info.append({
                "id": i,
                "left": monitor["left"],
                "top": monitor["top"], 
                "width": monitor["width"],
                "height": monitor["height"],
                "is_primary": i == 1  # Assume first monitor is primary
            })
        return monitors_info
    
    def capture_full_screen(self, monitor_id: int = 1) -> CaptureResult:
        """
        Capture full screen from specified monitor
        
        Args:
            monitor_id: Monitor ID (1 for primary, 0 for all monitors)
        """
        if monitor_id >= len(self.monitors):
            raise ValueError(f"Monitor {monitor_id} not found")
        
        start_time = time.time()
        
        # Capture screenshot
        monitor = self.monitors[monitor_id]
        screenshot = self.sct.grab(monitor)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Convert to bytes
        img_byte_buffer = BytesIO()
        img.save(img_byte_buffer, format='PNG', optimize=True)
        image_data = img_byte_buffer.getvalue()
        
        # Calculate hash
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        capture_time = time.time() - start_time
        
        return CaptureResult(
            image_data=image_data,
            image_hash=image_hash,
            dimensions=(screenshot.width, screenshot.height),
            monitor_info=monitor.copy(),
            capture_time=capture_time,
            file_size=len(image_data)
        )
    
    def capture_region(self, region: ScreenRegion) -> CaptureResult:
        """
        Capture specific screen region
        
        Args:
            region: Screen region to capture
        """
        start_time = time.time()
        
        # Capture screenshot of region
        screenshot = self.sct.grab(region.to_mss_dict())
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Convert to bytes
        img_byte_buffer = BytesIO()
        img.save(img_byte_buffer, format='PNG', optimize=True)
        image_data = img_byte_buffer.getvalue()
        
        # Calculate hash
        image_hash = hashlib.sha256(image_data).hexdigest()
        
        capture_time = time.time() - start_time
        
        return CaptureResult(
            image_data=image_data,
            image_hash=image_hash,
            dimensions=(screenshot.width, screenshot.height),
            monitor_info=region.to_mss_dict(),
            capture_time=capture_time,
            file_size=len(image_data)
        )
    
    def has_screen_changed(self, monitor_id: int = 1, 
                          sample_regions: Optional[List[ScreenRegion]] = None) -> bool:
        """
        Check if screen content has changed significantly
        
        Args:
            monitor_id: Monitor to check
            sample_regions: Specific regions to sample for changes (if None, uses full screen)
        
        Returns:
            True if screen has changed beyond threshold
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_capture_time < self.min_capture_interval:
            return False
        
        try:
            if sample_regions:
                # Capture multiple small regions for change detection
                current_images = []
                for region in sample_regions:
                    result = self.capture_region(region)
                    img = Image.open(BytesIO(result.image_data))
                    current_images.append(np.array(img.convert('L')))  # Convert to grayscale
                
                current_gray = np.concatenate([img.flatten() for img in current_images])
            else:
                # Capture full screen at reduced resolution for change detection
                monitor = self.monitors[monitor_id]
                
                # Capture at 1/4 resolution for faster processing
                reduced_monitor = {
                    "left": monitor["left"],
                    "top": monitor["top"],
                    "width": monitor["width"] // 4,
                    "height": monitor["height"] // 4
                }
                
                screenshot = self.sct.grab(reduced_monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                current_gray = np.array(img.convert('L'))
            
            self.last_capture_time = current_time
            
            # Compare with last image if available
            if self.last_image_gray is not None:
                # Calculate difference
                diff = cv2.absdiff(self.last_image_gray, current_gray)
                change_ratio = np.sum(diff > 30) / diff.size  # Threshold for "significant" pixel change
                
                logger.debug(f"Screen change ratio: {change_ratio:.3f}")
                
                has_changed = change_ratio > self.change_threshold
                
                if has_changed:
                    self.last_image_gray = current_gray
                
                return has_changed
            else:
                # First capture
                self.last_image_gray = current_gray
                return True
                
        except Exception as e:
            logger.error(f"Error checking screen changes: {e}")
            return True  # Assume changed on error
    
    def start_monitoring(self, callback, monitor_id: int = 1, 
                        check_interval: float = 0.5) -> None:
        """
        Start continuous screen monitoring (blocking)
        
        Args:
            callback: Function to call when screen changes (receives CaptureResult)
            monitor_id: Monitor to watch
            check_interval: How often to check for changes (seconds)
        """
        logger.info(f"Starting screen monitoring on monitor {monitor_id}")
        
        try:
            while True:
                if self.has_screen_changed(monitor_id):
                    logger.debug("Screen change detected, capturing...")
                    result = self.capture_full_screen(monitor_id)
                    
                    # Only call callback if image is actually different
                    if result.image_hash != self.last_image_hash:
                        self.last_image_hash = result.image_hash
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Error in capture callback: {e}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            logger.info("Screen monitoring stopped by user")
        except Exception as e:
            logger.error(f"Screen monitoring error: {e}")
            raise
    
    def capture_window(self, window_title: str = None, 
                      window_class: str = None) -> Optional[CaptureResult]:
        """
        Capture specific window (platform-specific implementation needed)
        
        Args:
            window_title: Title of window to capture
            window_class: Class name of window to capture
        
        Returns:
            CaptureResult if window found, None otherwise
        """
        # This would require platform-specific window detection
        # For now, just capture full screen
        logger.warning("Window-specific capture not implemented, using full screen")
        return self.capture_full_screen()
    
    def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position"""
        try:
            import pyautogui
            return pyautogui.position()
        except ImportError:
            logger.warning("pyautogui not available for cursor position")
            return (0, 0)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'sct'):
            self.sct.close()
            logger.info("Screen monitor cleaned up")


# Utility functions for common use cases
def quick_capture(monitor_id: int = 1) -> CaptureResult:
    """Quick screen capture without monitoring setup"""
    monitor = ScreenMonitor()
    try:
        return monitor.capture_full_screen(monitor_id)
    finally:
        monitor.cleanup()


def capture_primary_screen() -> CaptureResult:
    """Capture primary screen"""
    return quick_capture(monitor_id=1)


def get_screen_info() -> List[Dict[str, Any]]:
    """Get information about available screens"""
    monitor = ScreenMonitor()
    try:
        return monitor.get_monitors()
    finally:
        monitor.cleanup()