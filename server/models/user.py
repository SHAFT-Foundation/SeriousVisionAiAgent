"""
User model for Vision Agent
Stores user profiles, accessibility preferences, and usage analytics
"""
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Boolean, Integer, Float, JSON, Text
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
    __tablename__ = "users"
    
    # Basic user information
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    full_name = Column(String(200), nullable=True)
    
    # Authentication (for future use with user accounts)
    hashed_password = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Accessibility Profile
    screen_reader_type = Column(String(50), nullable=True)  # NVDA, JAWS, VoiceOver, TalkBack
    uses_braille_display = Column(Boolean, default=False)
    motor_limitations = Column(JSON, nullable=True)  # Switch control, eye tracking, etc.
    visual_impairment_type = Column(String(100), nullable=True)  # blind, low_vision, cortical, etc.
    
    # Processing Preferences
    default_verbosity = Column(String(20), default="medium")  # brief, medium, detailed
    preferred_voice_settings = Column(JSON, nullable=True)  # Speed, pitch, voice type
    domain_specific_settings = Column(JSON, nullable=True)  # Different settings per context
    output_preferences = Column(JSON, nullable=True)  # text, audio, braille priorities
    
    # Privacy Settings
    data_retention_days = Column(Integer, default=30)
    allow_cloud_processing = Column(Boolean, default=True)
    require_local_processing_for_pii = Column(Boolean, default=True)
    
    # Usage Analytics and Learning
    total_processing_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    average_session_duration = Column(Float, nullable=True)  # in minutes
    most_used_contexts = Column(JSON, nullable=True)  # frequency count by domain
    preferred_llm_providers = Column(JSON, nullable=True)  # provider preference scores
    
    # Quality and Feedback
    average_satisfaction_rating = Column(Float, nullable=True)  # 1-5 scale
    total_feedback_submissions = Column(Integer, default=0)
    
    # System preferences
    max_concurrent_requests = Column(Integer, default=3)
    cache_preferences = Column(JSON, nullable=True)
    
    # Relationships
    processing_jobs = relationship("ProcessingJob", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', screen_reader='{self.screen_reader_type}')>"
    
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate"""
        if self.total_processing_requests == 0:
            return 0.0
        return self.successful_requests / self.total_processing_requests
    
    def update_usage_stats(self, success: bool = True, context: str = "general") -> None:
        """Update user usage statistics"""
        self.total_processing_requests += 1
        if success:
            self.successful_requests += 1
        
        # Update context usage
        if self.most_used_contexts is None:
            self.most_used_contexts = {}
        
        self.most_used_contexts[context] = self.most_used_contexts.get(context, 0) + 1
    
    def get_preferred_settings_for_context(self, context: str) -> Dict[str, Any]:
        """Get user preferences for specific context"""
        base_settings = {
            "verbosity": self.default_verbosity,
            "voice_settings": self.preferred_voice_settings or {},
            "output_preferences": self.output_preferences or {"primary": "text"}
        }
        
        # Override with context-specific settings if available
        if self.domain_specific_settings and context in self.domain_specific_settings:
            base_settings.update(self.domain_specific_settings[context])
        
        return base_settings
    
    def can_use_cloud_processing(self, contains_pii: bool = False) -> bool:
        """Determine if cloud processing is allowed based on user preferences"""
        if not self.allow_cloud_processing:
            return False
        
        if contains_pii and self.require_local_processing_for_pii:
            return False
        
        return True