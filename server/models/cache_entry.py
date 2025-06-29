"""
Cache entry model for Vision Agent
Stores cached processing results for performance optimization
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Integer, JSON, DateTime, Boolean, Index
from .base import Base


class CacheEntry(Base):
    __tablename__ = "cache_entries"
    
    # Cache identification
    cache_key = Column(String(100), unique=True, nullable=False, index=True)
    image_hash = Column(String(64), nullable=False, index=True)  # SHA-256 of original image
    
    # Cache metadata
    hit_count = Column(Integer, default=1)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    cache_size_bytes = Column(Integer, nullable=True)
    
    # Processing context that generated this cache
    processing_context = Column(String(50), nullable=False, index=True)
    verbosity_level = Column(String(20), nullable=False)
    llm_provider = Column(String(30), nullable=False)
    model_used = Column(String(100), nullable=True)
    
    # Cached results
    accessibility_result = Column(JSON, nullable=False)  # Complete accessibility annotation
    processing_metadata = Column(JSON, nullable=True)    # Original processing metadata
    
    # Cache management
    expires_at = Column(DateTime, nullable=True, index=True)
    is_permanent = Column(Boolean, default=False)  # Some results can be marked as permanent
    invalidated = Column(Boolean, default=False, index=True)
    
    # Quality tracking
    confidence_score = Column(Integer, nullable=True)  # Original confidence * 100 for indexing
    user_validated = Column(Boolean, default=False)
    validation_score = Column(Integer, nullable=True)  # User rating * 20 for 1-5 scale
    
    # Add composite indexes for common queries
    __table_args__ = (
        Index('idx_cache_context_verbosity', 'processing_context', 'verbosity_level'),
        Index('idx_cache_hash_context', 'image_hash', 'processing_context'),
        Index('idx_cache_expires_invalidated', 'expires_at', 'invalidated'),
    )
    
    def __repr__(self) -> str:
        return f"<CacheEntry(key='{self.cache_key}', hits={self.hit_count}, expires={self.expires_at})>"
    
    @classmethod
    def generate_cache_key(cls, image_hash: str, context: str, verbosity: str, 
                          provider: str, model: str = None) -> str:
        """Generate a consistent cache key from parameters"""
        key_parts = [image_hash[:16], context, verbosity, provider]
        if model:
            key_parts.append(model.split('/')[-1])  # Use just the model name, not full path
        
        return "_".join(key_parts)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.is_permanent:
            return False
        if self.invalidated:
            return True
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    @property
    def is_valid(self) -> bool:
        """Check if cache entry is valid for use"""
        return not self.is_expired and not self.invalidated
    
    def access(self) -> None:
        """Record cache access"""
        self.hit_count += 1
        self.last_accessed = datetime.utcnow()
    
    def set_expiration(self, hours: int = 24) -> None:
        """Set cache expiration time"""
        if not self.is_permanent:
            self.expires_at = datetime.utcnow() + timedelta(hours=hours)
    
    def invalidate(self) -> None:
        """Mark cache entry as invalid"""
        self.invalidated = True
    
    def make_permanent(self) -> None:
        """Mark cache entry as permanent (never expires)"""
        self.is_permanent = True
        self.expires_at = None
    
    def add_user_validation(self, user_rating: float) -> None:
        """Add user validation score"""
        self.user_validated = True
        self.validation_score = int(user_rating * 20)  # Convert 1-5 scale to 20-100
    
    def get_accessibility_result(self) -> Dict[str, Any]:
        """Get the cached accessibility result"""
        return self.accessibility_result or {}
    
    def should_refresh(self, max_age_hours: int = 168) -> bool:  # Default 1 week
        """Check if cache should be refreshed based on age"""
        if self.is_permanent:
            return False
        
        age = datetime.utcnow() - self.created_at
        return age > timedelta(hours=max_age_hours)
    
    @classmethod
    def cleanup_expired(cls, session) -> int:
        """Clean up expired cache entries and return count deleted"""
        now = datetime.utcnow()
        
        # Delete expired entries (but keep permanent ones)
        deleted_count = session.query(cls).filter(
            cls.expires_at < now,
            cls.is_permanent == False
        ).delete()
        
        # Also delete invalidated entries older than 24 hours
        yesterday = now - timedelta(hours=24)
        deleted_count += session.query(cls).filter(
            cls.invalidated == True,
            cls.updated_at < yesterday
        ).delete()
        
        session.commit()
        return deleted_count
    
    @classmethod
    def get_cache_stats(cls, session) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = session.query(cls).count()
        valid_entries = session.query(cls).filter(
            cls.invalidated == False,
            (cls.expires_at > datetime.utcnow()) | (cls.expires_at.is_(None))
        ).count()
        
        total_hits = session.query(cls).with_entities(
            cls.hit_count
        ).scalar() or 0
        
        avg_confidence = session.query(cls).with_entities(
            cls.confidence_score
        ).filter(cls.confidence_score.isnot(None)).scalar()
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "total_cache_hits": total_hits,
            "cache_efficiency": valid_entries / total_entries if total_entries > 0 else 0,
            "average_confidence": avg_confidence / 100.0 if avg_confidence else None
        }