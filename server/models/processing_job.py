"""
Processing job model for Vision Agent
Tracks image processing requests and their status
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Integer, Float, JSON, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .base import Base


class ProcessingStatus(str, Enum):
    """Processing job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    CANCELLED = "cancelled"


class CaptureSource(str, Enum):
    """Source of the captured image"""
    SCREEN = "screen"
    CAMERA = "camera"
    UPLOAD = "upload"
    CLIPBOARD = "clipboard"


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="processing_jobs")
    
    # Image Information
    image_hash = Column(String(64), nullable=False, index=True)  # SHA-256 for deduplication
    image_dimensions = Column(JSON, nullable=True)  # {"width": 1920, "height": 1080}
    image_size_bytes = Column(Integer, nullable=True)
    capture_source = Column(String(20), nullable=False, default=CaptureSource.SCREEN)
    capture_metadata = Column(JSON, nullable=True)  # Monitor info, camera settings, etc.
    
    # Processing Configuration
    processing_context = Column(String(50), default="general")  # code, academic, business, personal
    verbosity_level = Column(String(20), default="medium")  # brief, medium, detailed
    requested_output_format = Column(String(20), default="text")  # text, audio, braille
    
    # Processing Details
    processing_status = Column(String(20), default=ProcessingStatus.QUEUED, index=True)
    llm_provider = Column(String(30), nullable=True)  # openai, anthropic, google, local
    model_used = Column(String(100), nullable=True)  # specific model version
    processing_start_time = Column(DateTime, nullable=True)
    processing_end_time = Column(DateTime, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # API Usage and Costs
    tokens_used = Column(Integer, nullable=True)
    api_cost_cents = Column(Float, nullable=True)  # Cost in cents
    
    # Results and Caching
    result_cache_key = Column(String(100), nullable=True, index=True)
    confidence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    was_cached_result = Column(Boolean, default=False)
    
    # Error Handling
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Quality and Feedback
    user_feedback_rating = Column(Float, nullable=True)  # 1-5 scale
    user_feedback_text = Column(Text, nullable=True)
    
    # Privacy and Security
    contains_pii = Column(Boolean, default=False)  # Detected personally identifiable information
    processed_locally = Column(Boolean, default=False)
    
    # Relationships
    accessibility_annotation = relationship("AccessibilityAnnotation", back_populates="processing_job", 
                                          uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, status='{self.processing_status}', provider='{self.llm_provider}')>"
    
    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Calculate processing duration in seconds"""
        if self.processing_time_ms:
            return self.processing_time_ms / 1000.0
        elif self.processing_start_time and self.processing_end_time:
            delta = self.processing_end_time - self.processing_start_time
            return delta.total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed (successfully or failed)"""
        return self.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CACHED]
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return self.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.CACHED]
    
    def start_processing(self, provider: str, model: str) -> None:
        """Mark job as started"""
        self.processing_status = ProcessingStatus.PROCESSING
        self.llm_provider = provider
        self.model_used = model
        self.processing_start_time = datetime.utcnow()
    
    def complete_processing(self, confidence_score: float, tokens_used: int = None, 
                          api_cost_cents: float = None, was_cached: bool = False) -> None:
        """Mark job as completed successfully"""
        self.processing_status = ProcessingStatus.CACHED if was_cached else ProcessingStatus.COMPLETED
        self.processing_end_time = datetime.utcnow()
        self.confidence_score = confidence_score
        self.was_cached_result = was_cached
        
        if tokens_used:
            self.tokens_used = tokens_used
        if api_cost_cents:
            self.api_cost_cents = api_cost_cents
        
        # Calculate processing time
        if self.processing_start_time:
            delta = self.processing_end_time - self.processing_start_time
            self.processing_time_ms = int(delta.total_seconds() * 1000)
    
    def fail_processing(self, error_message: str, error_code: str = None) -> None:
        """Mark job as failed"""
        self.processing_status = ProcessingStatus.FAILED
        self.processing_end_time = datetime.utcnow()
        self.error_message = error_message
        self.error_code = error_code
        
        if self.processing_start_time:
            delta = self.processing_end_time - self.processing_start_time
            self.processing_time_ms = int(delta.total_seconds() * 1000)
    
    def add_user_feedback(self, rating: float, feedback_text: str = None) -> None:
        """Add user feedback for this processing job"""
        self.user_feedback_rating = max(1.0, min(5.0, rating))  # Clamp to 1-5 range
        self.user_feedback_text = feedback_text
    
    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if job can be retried"""
        return (self.processing_status == ProcessingStatus.FAILED and 
                self.retry_count < max_retries)
    
    def increment_retry(self) -> None:
        """Increment retry count and reset to queued status"""
        self.retry_count += 1
        self.processing_status = ProcessingStatus.QUEUED
        self.error_message = None
        self.error_code = None
        self.processing_start_time = None
        self.processing_end_time = None