"""
User management endpoints for Vision Agent API
"""
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from ..utils.database import get_db
from ..models import User, ProcessingJob

router = APIRouter()
logger = logging.getLogger(__name__)


class UserPreferences(BaseModel):
    """User accessibility preferences"""
    screen_reader_type: Optional[str] = Field(None, description="Screen reader software")
    uses_braille_display: bool = Field(False, description="Uses braille display")
    default_verbosity: str = Field("medium", description="Preferred detail level")
    preferred_voice_settings: Optional[Dict[str, Any]] = Field(None, description="TTS settings")
    domain_specific_settings: Optional[Dict[str, Any]] = Field(None, description="Context-specific settings")
    allow_cloud_processing: bool = Field(True, description="Allow cloud processing")
    require_local_processing_for_pii: bool = Field(True, description="Force local processing for PII")
    max_concurrent_requests: int = Field(3, description="Max concurrent processing requests")


class UserProfile(BaseModel):
    """User profile response"""
    id: str
    username: str
    email: Optional[str]
    full_name: Optional[str]
    screen_reader_type: Optional[str]
    uses_braille_display: bool
    default_verbosity: str
    total_processing_requests: int
    successful_requests: int
    success_rate: float
    average_satisfaction_rating: Optional[float]
    created_at: str
    last_active: str


class UserStats(BaseModel):
    """User usage statistics"""
    total_requests: int
    successful_requests: int
    success_rate: float
    most_used_contexts: Dict[str, int]
    average_processing_time_ms: Optional[float]
    average_satisfaction_rating: Optional[float]
    total_feedback_submissions: int
    recent_activity: List[Dict[str, Any]]


@router.get("/users/{user_id}", response_model=UserProfile)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db)
) -> UserProfile:
    """Get user profile by ID"""
    try:
        result = await db.execute(
            select(User).where(User.username == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserProfile(
            id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            screen_reader_type=user.screen_reader_type,
            uses_braille_display=user.uses_braille_display,
            default_verbosity=user.default_verbosity,
            total_processing_requests=user.total_processing_requests,
            successful_requests=user.successful_requests,
            success_rate=user.success_rate,
            average_satisfaction_rating=user.average_satisfaction_rating,
            created_at=user.created_at.isoformat(),
            last_active=user.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving user")


@router.get("/users/{user_id}/preferences", response_model=UserPreferences)
async def get_user_preferences(
    user_id: str,
    db: AsyncSession = Depends(get_db)
) -> UserPreferences:
    """Get user accessibility preferences"""
    try:
        result = await db.execute(
            select(User).where(User.username == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserPreferences(
            screen_reader_type=user.screen_reader_type,
            uses_braille_display=user.uses_braille_display,
            default_verbosity=user.default_verbosity,
            preferred_voice_settings=user.preferred_voice_settings,
            domain_specific_settings=user.domain_specific_settings,
            allow_cloud_processing=user.allow_cloud_processing,
            require_local_processing_for_pii=user.require_local_processing_for_pii,
            max_concurrent_requests=user.max_concurrent_requests
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving preferences for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving preferences")


@router.post("/users/{user_id}/preferences")
async def update_user_preferences(
    user_id: str,
    preferences: UserPreferences,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Update user accessibility preferences"""
    try:
        result = await db.execute(
            select(User).where(User.username == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user if doesn't exist
            user = User(username=user_id)
            db.add(user)
        
        # Update preferences
        user.screen_reader_type = preferences.screen_reader_type
        user.uses_braille_display = preferences.uses_braille_display
        user.default_verbosity = preferences.default_verbosity
        user.preferred_voice_settings = preferences.preferred_voice_settings
        user.domain_specific_settings = preferences.domain_specific_settings
        user.allow_cloud_processing = preferences.allow_cloud_processing
        user.require_local_processing_for_pii = preferences.require_local_processing_for_pii
        user.max_concurrent_requests = preferences.max_concurrent_requests
        
        await db.commit()
        
        return {"status": "success", "message": "Preferences updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error updating preferences")


@router.get("/users/{user_id}/stats", response_model=UserStats)
async def get_user_stats(
    user_id: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
) -> UserStats:
    """Get user usage statistics"""
    try:
        result = await db.execute(
            select(User).where(User.username == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get recent processing jobs
        recent_jobs_result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.user_id == user.id)
            .order_by(ProcessingJob.created_at.desc())
            .limit(limit)
        )
        recent_jobs = recent_jobs_result.scalars().all()
        
        # Calculate average processing time
        completed_jobs = [j for j in recent_jobs if j.processing_time_ms]
        avg_processing_time = None
        if completed_jobs:
            avg_processing_time = sum(j.processing_time_ms for j in completed_jobs) / len(completed_jobs)
        
        # Format recent activity
        recent_activity = []
        for job in recent_jobs:
            recent_activity.append({
                "job_id": str(job.id),
                "status": job.processing_status,
                "context": job.processing_context,
                "created_at": job.created_at.isoformat(),
                "processing_time_ms": job.processing_time_ms,
                "confidence_score": job.confidence_score
            })
        
        return UserStats(
            total_requests=user.total_processing_requests,
            successful_requests=user.successful_requests,
            success_rate=user.success_rate,
            most_used_contexts=user.most_used_contexts or {},
            average_processing_time_ms=avg_processing_time,
            average_satisfaction_rating=user.average_satisfaction_rating,
            total_feedback_submissions=user.total_feedback_submissions,
            recent_activity=recent_activity
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stats for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving stats")


@router.post("/users/{user_id}/feedback")
async def submit_feedback(
    user_id: str,
    job_id: str,
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating from 1-5"),
    feedback_text: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Submit user feedback for a processing job"""
    try:
        # Find the processing job
        job = await db.get(ProcessingJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Processing job not found")
        
        # Verify job belongs to user
        result = await db.execute(
            select(User).where(User.username == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user or job.user_id != user.id:
            raise HTTPException(status_code=403, detail="Job does not belong to user")
        
        # Add feedback to job
        job.add_user_feedback(rating, feedback_text)
        
        # Update user statistics
        user.total_feedback_submissions += 1
        
        # Recalculate average satisfaction rating
        avg_result = await db.execute(
            select(ProcessingJob.user_feedback_rating)
            .where(ProcessingJob.user_id == user.id)
            .where(ProcessingJob.user_feedback_rating.isnot(None))
        )
        ratings = [r[0] for r in avg_result.fetchall()]
        if ratings:
            user.average_satisfaction_rating = sum(ratings) / len(ratings)
        
        await db.commit()
        
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Error submitting feedback")