"""Health check endpoints for monitoring."""

from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import DB
from app.core.config import settings
from app.services.gpu_status_service import gpu_status_service
from app.utils.envelopes import api_success

router = APIRouter(tags=["health"])


@router.get("/health", response_model=dict)
async def health_check(db: DB):
    """Health check endpoint for load balancers and monitoring."""
    # Test database connectivity
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    health_data = {
        "status": "ok" if db_status == "healthy" else "degraded",
        "service": settings.APP_NAME,
        "database": db_status,
    }

    return api_success(health_data)


@router.get("/health/ready", response_model=dict)
async def readiness_check(db: DB):
    """Kubernetes readiness probe."""
    try:
        await db.execute(text("SELECT 1"))
        return api_success({"ready": True})
    except Exception:
        return api_success({"ready": False})


@router.get("/health/live", response_model=dict)
async def liveness_check():
    """Kubernetes liveness probe."""
    return api_success({"alive": True})


@router.get("/health/3d-machine", response_model=dict)
async def check_3d_machine(db: DB):
    """Fast 3D machine health/estimate endpoint for frontend workflows."""
    result = await gpu_status_service.get_status(db=db, touch_activation=True)
    return api_success(result)
