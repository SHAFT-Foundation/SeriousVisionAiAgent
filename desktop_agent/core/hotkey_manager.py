"""
Global hotkey management for Vision Agent
"""
import logging
import threading
from typing import Dict, Callable, Optional, Set, List
from dataclasses import dataclass
from enum import Enum
import time
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

logger = logging.getLogger(__name__)


class HotkeyAction(Enum):
    """Predefined hotkey actions"""
    CAPTURE_SCREEN = "capture_screen"
    CAPTURE_REGION = "capture_region"
    REPEAT_LAST = "repeat_last"
    TOGGLE_MONITORING = "toggle_monitoring"
    STOP_SPEECH = "stop_speech"
    PAUSE_SPEECH = "pause_speech"
    INCREASE_VERBOSITY = "increase_verbosity"
    DECREASE_VERBOSITY = "decrease_verbosity"
    TOGGLE_TTS = "toggle_tts"
    SHOW_HELP = "show_help"
    QUIT_APP = "quit_app"


@dataclass
class HotkeyDefinition:
    """Definition of a hotkey combination"""
    keys: Set[Key]
    action: HotkeyAction
    callback: Callable
    description: str
    enabled: bool = True
    
    def __hash__(self):
        return hash(tuple(sorted(self.keys, key=str)))


class HotkeyManager:
    """Manages global hotkeys for accessibility"""
    
    def __init__(self):
        """Initialize hotkey manager"""
        self.hotkeys: Dict[frozenset, HotkeyDefinition] = {}
        self.pressed_keys: Set[Key] = set()
        self.listener: Optional[keyboard.Listener] = None
        self.enabled = False
        
        # Callback registry
        self.action_callbacks: Dict[HotkeyAction, Callable] = {}
        
        # Rate limiting
        self.last_trigger_time = {}
        self.min_trigger_interval = 0.5  # Minimum seconds between triggers
        
        # Default hotkey combinations
        self._setup_default_hotkeys()
        
        logger.info("Hotkey manager initialized")
    
    def _setup_default_hotkeys(self):
        """Setup default hotkey combinations"""
        default_hotkeys = [
            # Primary capture
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('c')},
                HotkeyAction.CAPTURE_SCREEN,
                "Capture and analyze current screen"
            ),
            
            # Region capture
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('r')},
                HotkeyAction.CAPTURE_REGION,
                "Capture and analyze selected region"
            ),
            
            # Repeat last result
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('l')},
                HotkeyAction.REPEAT_LAST,
                "Repeat last accessibility result"
            ),
            
            # Toggle monitoring
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('m')},
                HotkeyAction.TOGGLE_MONITORING,
                "Toggle automatic screen monitoring"
            ),
            
            # Speech controls
            (
                {Key.ctrl, Key.alt, Key.space},
                HotkeyAction.STOP_SPEECH,
                "Stop current speech"
            ),
            
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('p')},
                HotkeyAction.PAUSE_SPEECH,
                "Pause/resume speech"
            ),
            
            # Verbosity controls
            (
                {Key.ctrl, Key.alt, Key.up},
                HotkeyAction.INCREASE_VERBOSITY,
                "Increase description detail level"
            ),
            
            (
                {Key.ctrl, Key.alt, Key.down},
                HotkeyAction.DECREASE_VERBOSITY,
                "Decrease description detail level"
            ),
            
            # TTS toggle
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('t')},
                HotkeyAction.TOGGLE_TTS,
                "Toggle text-to-speech on/off"
            ),
            
            # Help and quit
            (
                {Key.ctrl, Key.alt, Key.f1},
                HotkeyAction.SHOW_HELP,
                "Show help and hotkey list"
            ),
            
            (
                {Key.ctrl, Key.alt, KeyCode.from_char('q')},
                HotkeyAction.QUIT_APP,
                "Quit Vision Agent"
            )
        ]
        
        # Register default hotkeys (callbacks will be set later)
        for keys, action, description in default_hotkeys:
            self.register_hotkey(keys, action, None, description)
    
    def register_hotkey(self, keys: Set[Key], action: HotkeyAction, 
                       callback: Optional[Callable] = None, 
                       description: str = "") -> bool:
        """
        Register a new hotkey combination
        
        Args:
            keys: Set of keys that must be pressed together
            action: Action identifier
            callback: Function to call when triggered
            description: Human-readable description
        
        Returns:
            True if registration successful
        """
        try:
            key_set = frozenset(keys)
            
            hotkey_def = HotkeyDefinition(
                keys=keys,
                action=action,
                callback=callback,
                description=description
            )
            
            self.hotkeys[key_set] = hotkey_def
            
            # Initialize rate limiting
            self.last_trigger_time[action] = 0
            
            logger.info(f"Registered hotkey: {self._format_keys(keys)} -> {action.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register hotkey {keys}: {e}")
            return False
    
    def unregister_hotkey(self, keys: Set[Key]) -> bool:
        """Unregister a hotkey combination"""
        try:
            key_set = frozenset(keys)
            if key_set in self.hotkeys:
                del self.hotkeys[key_set]
                logger.info(f"Unregistered hotkey: {self._format_keys(keys)}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unregister hotkey {keys}: {e}")
            return False
    
    def register_action_callback(self, action: HotkeyAction, callback: Callable):
        """Register callback for a specific action"""
        self.action_callbacks[action] = callback
        logger.info(f"Registered callback for action: {action.value}")
    
    def enable_hotkey(self, keys: Set[Key], enabled: bool = True):
        """Enable or disable specific hotkey"""
        key_set = frozenset(keys)
        if key_set in self.hotkeys:
            self.hotkeys[key_set].enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Hotkey {self._format_keys(keys)} {status}")
    
    def start_listening(self) -> bool:
        """Start listening for global hotkeys"""
        if self.listener and self.listener.running:
            return True
        
        try:
            self.listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
                suppress=False  # Don't suppress keys, just listen
            )
            
            self.listener.start()
            self.enabled = True
            
            logger.info("Hotkey listener started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start hotkey listener: {e}")
            return False
    
    def stop_listening(self):
        """Stop listening for hotkeys"""
        self.enabled = False
        
        if self.listener:
            self.listener.stop()
            self.listener = None
        
        self.pressed_keys.clear()
        logger.info("Hotkey listener stopped")
    
    def _on_key_press(self, key):
        """Handle key press events"""
        if not self.enabled:
            return
        
        try:
            # Normalize key
            normalized_key = self._normalize_key(key)
            if normalized_key:
                self.pressed_keys.add(normalized_key)
                
                # Check for hotkey matches
                self._check_hotkey_trigger()
                
        except Exception as e:
            logger.error(f"Error handling key press {key}: {e}")
    
    def _on_key_release(self, key):
        """Handle key release events"""
        if not self.enabled:
            return
        
        try:
            # Normalize key
            normalized_key = self._normalize_key(key)
            if normalized_key and normalized_key in self.pressed_keys:
                self.pressed_keys.remove(normalized_key)
                
        except Exception as e:
            logger.error(f"Error handling key release {key}: {e}")
    
    def _normalize_key(self, key) -> Optional[Key]:
        """Normalize key to consistent format"""
        try:
            # Handle special keys
            if hasattr(key, 'vk') or isinstance(key, Key):
                return key
            
            # Handle character keys
            if hasattr(key, 'char') and key.char:
                return KeyCode.from_char(key.char.lower())
            
            return key
            
        except Exception:
            return None
    
    def _check_hotkey_trigger(self):
        """Check if current pressed keys match any registered hotkey"""
        current_keys = frozenset(self.pressed_keys)
        
        for key_set, hotkey_def in self.hotkeys.items():
            if not hotkey_def.enabled:
                continue
            
            # Check if all required keys are pressed
            if key_set.issubset(current_keys) and len(key_set) == len(current_keys):
                self._trigger_hotkey(hotkey_def)
                break
    
    def _trigger_hotkey(self, hotkey_def: HotkeyDefinition):
        """Trigger a hotkey action"""
        try:
            # Rate limiting
            current_time = time.time()
            last_time = self.last_trigger_time.get(hotkey_def.action, 0)
            
            if current_time - last_time < self.min_trigger_interval:
                return
            
            self.last_trigger_time[hotkey_def.action] = current_time
            
            logger.info(f"Hotkey triggered: {hotkey_def.action.value}")
            
            # Call specific callback first
            if hotkey_def.callback:
                threading.Thread(
                    target=hotkey_def.callback,
                    daemon=True
                ).start()
            
            # Call action callback
            if hotkey_def.action in self.action_callbacks:
                callback = self.action_callbacks[hotkey_def.action]
                threading.Thread(
                    target=callback,
                    daemon=True
                ).start()
            
        except Exception as e:
            logger.error(f"Error triggering hotkey {hotkey_def.action}: {e}")
    
    def _format_keys(self, keys: Set[Key]) -> str:
        """Format key combination for display"""
        key_names = []
        
        for key in sorted(keys, key=str):
            if isinstance(key, Key):
                # Special keys
                name_map = {
                    Key.ctrl: "Ctrl",
                    Key.alt: "Alt", 
                    Key.shift: "Shift",
                    Key.cmd: "Cmd",
                    Key.space: "Space",
                    Key.enter: "Enter",
                    Key.tab: "Tab",
                    Key.up: "↑",
                    Key.down: "↓",
                    Key.left: "←",
                    Key.right: "→",
                    Key.f1: "F1",
                    Key.f2: "F2",
                    Key.f3: "F3",
                    Key.f4: "F4",
                    Key.f5: "F5",
                    Key.f6: "F6",
                    Key.f7: "F7",
                    Key.f8: "F8",
                    Key.f9: "F9",
                    Key.f10: "F10",
                    Key.f11: "F11",
                    Key.f12: "F12"
                }
                key_names.append(name_map.get(key, str(key)))
            elif hasattr(key, 'char') and key.char:
                key_names.append(key.char.upper())
            else:
                key_names.append(str(key))
        
        return " + ".join(key_names)
    
    def get_hotkey_list(self) -> List[Dict[str, str]]:
        """Get list of all registered hotkeys"""
        hotkey_list = []
        
        for key_set, hotkey_def in self.hotkeys.items():
            hotkey_list.append({
                "keys": self._format_keys(hotkey_def.keys),
                "action": hotkey_def.action.value,
                "description": hotkey_def.description,
                "enabled": hotkey_def.enabled
            })
        
        return sorted(hotkey_list, key=lambda x: x["keys"])
    
    def get_help_text(self) -> str:
        """Get formatted help text for all hotkeys"""
        lines = ["Vision Agent Hotkeys:", "=" * 25, ""]
        
        for hotkey_info in self.get_hotkey_list():
            if hotkey_info["enabled"]:
                lines.append(f"{hotkey_info['keys']:<20} {hotkey_info['description']}")
        
        return "\n".join(lines)
    
    def is_listening(self) -> bool:
        """Check if hotkey listener is active"""
        return self.enabled and self.listener and self.listener.running


# Global hotkey manager instance
_global_hotkey_manager: Optional[HotkeyManager] = None

def get_hotkey_manager() -> HotkeyManager:
    """Get global hotkey manager instance"""
    global _global_hotkey_manager
    if _global_hotkey_manager is None:
        _global_hotkey_manager = HotkeyManager()
    return _global_hotkey_manager