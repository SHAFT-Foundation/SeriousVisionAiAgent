"""
Image processing endpoints for Vision Agent API
"""
import base64
import hashlib
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from ..utils.database import get_db
from ..models import User, ProcessingJob, AccessibilityAnnotation, CaptureSource
from ..utils.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


class ProcessingRequest(BaseModel):
    """Request model for image processing"""
    image_data: str = Field(..., description="Base64 encoded image")
    user_id: str = Field(..., description="User identifier") 
    context: Optional[str] = Field("general", description="Processing context")
    verbosity: str = Field("medium", description="Detail level: brief, medium, detailed")
    output_format: str = Field("text", description="Output format: text, audio, braille")
    force_reprocess: bool = Field(False, description="Skip cache and reprocess")


class ProcessingResponse(BaseModel):
    """Response model for processed image"""
    job_id: str = Field(..., description="Processing job ID")
    status: str = Field(..., description="Processing status")
    alt_text: str = Field(..., description="Primary alternative text")
    detailed_description: Optional[str] = Field(None, description="Detailed description")
    structural_elements: Dict[str, Any] = Field(default_factory=dict)
    interactive_elements: list = Field(default_factory=list)
    reading_order: list = Field(default_factory=list)
    confidence_score: float = Field(..., description="Processing confidence")
    processing_time_ms: int = Field(..., description="Processing duration")
    provider_used: str = Field(..., description="LLM provider used")
    was_cached: bool = Field(False, description="Result served from cache")


@router.post("/process", response_model=ProcessingResponse)
async def process_image(
    request: ProcessingRequest,
    db: AsyncSession = Depends(get_db)
) -> ProcessingResponse:
    """Process an image and return accessibility metadata"""
    try:
        # Validate and decode image
        try:
            image_bytes = base64.b64decode(request.image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Calculate image hash for caching and deduplication
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Get or create user
        user = await _get_or_create_user(db, request.user_id)
        
        # Create processing job
        job = ProcessingJob(
            user_id=user.id,
            image_hash=image_hash,
            image_size_bytes=len(image_bytes),
            capture_source=CaptureSource.UPLOAD,
            processing_context=request.context,
            verbosity_level=request.verbosity,
            requested_output_format=request.output_format
        )
        
        db.add(job)
        await db.commit()
        await db.refresh(job)
        
        # TODO: Check cache first if not force_reprocess
        # TODO: Queue job for processing
        # TODO: For now, return a mock response
        
        # Create mock accessibility annotation
        annotation = AccessibilityAnnotation(
            job_id=job.id,
            alt_text="This is a placeholder description. Image processing not yet implemented.",
            confidence_score=0.9,
            processing_model="mock-model"
        )
        
        db.add(annotation)
        job.complete_processing(confidence_score=0.9, was_cached=False)
        
        await db.commit()
        await db.refresh(annotation)
        
        return ProcessingResponse(
            job_id=str(job.id),
            status=job.processing_status,
            alt_text=annotation.alt_text,
            detailed_description=annotation.detailed_description,
            structural_elements=annotation.get_structural_elements(),
            interactive_elements=annotation.get_interactive_elements(),
            reading_order=annotation.reading_order or [],
            confidence_score=annotation.confidence_score,
            processing_time_ms=job.processing_time_ms or 0,
            provider_used=job.llm_provider or "mock",
            was_cached=job.was_cached_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


@router.post("/process/upload")
async def process_uploaded_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    context: str = Form("general"),
    verbosity: str = Form("medium"),
    output_format: str = Form("text"),
    db: AsyncSession = Depends(get_db)
) -> ProcessingResponse:
    """Process an uploaded image file"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and encode image
        image_bytes = await file.read()
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create processing request
        request = ProcessingRequest(
            image_data=image_data,
            user_id=user_id,
            context=context,
            verbosity=verbosity,
            output_format=output_format
        )
        
        return await process_image(request, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload processing error")


@router.get("/process/job/{job_id}")
async def get_processing_job(
    job_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get processing job status and results"""
    try:
        # Find processing job
        job = await db.get(ProcessingJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Processing job not found")
        
        result = {
            "job_id": str(job.id),
            "status": job.processing_status,
            "created_at": job.created_at.isoformat(),
            "processing_time_ms": job.processing_time_ms,
            "confidence_score": job.confidence_score,
            "provider_used": job.llm_provider,
            "context": job.processing_context,
            "verbosity": job.verbosity_level
        }
        
        # Add results if completed
        if job.accessibility_annotation:
            annotation = job.accessibility_annotation
            result.update({
                "alt_text": annotation.alt_text,
                "detailed_description": annotation.detailed_description,
                "structural_elements": annotation.get_structural_elements(),
                "interactive_elements": annotation.get_interactive_elements(),
                "reading_order": annotation.reading_order or []
            })
        
        # Add error info if failed
        if job.processing_status == "failed":
            result.update({
                "error_message": job.error_message,
                "error_code": job.error_code
            })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving job")


async def _get_or_create_user(db: AsyncSession, user_id: str) -> User:
    """Get existing user or create new one"""
    # Try to find existing user
    from sqlalchemy import select
    
    result = await db.execute(
        select(User).where(User.username == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user
        user = User(username=user_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    return user