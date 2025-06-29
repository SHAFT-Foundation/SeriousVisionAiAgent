"""
Health check endpoints for Vision Agent API
"""
import time
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..utils.database import get_db
from ..utils.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "vision-agent-api",
        "version": "0.1.0"
    }


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Detailed health check with system and database status"""
    settings = get_settings()
    start_time = time.time()
    
    # Test database connection
    db_status = "healthy"
    db_response_time = None
    try:
        db_start = time.time()
        result = await db.execute(text("SELECT 1"))
        db_response_time = round((time.time() - db_start) * 1000, 2)  # ms
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Get system metrics
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    response_time = round((time.time() - start_time) * 1000, 2)  # ms
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "vision-agent-api",
        "version": "0.1.0",
        "response_time_ms": response_time,
        "database": {
            "status": db_status,
            "response_time_ms": db_response_time
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "used_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            },
            "disk": {
                "used_percent": round((disk.used / disk.total) * 100, 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            }
        },
        "configuration": {
            "debug_mode": settings.debug,
            "max_image_size": settings.max_image_size,
            "max_concurrent_requests": settings.max_concurrent_requests
        }
    }


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint"""
    try:
        # Test database connection
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503


@router.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive"}