"""Master API router — mounts all sub-routers under /api/v1."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.routes.admin import router as admin_router
from backend.api.routes.auth import router as auth_router
from backend.api.routes.workflows import router as workflows_router
from backend.api.routes.outreach import router as outreach_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(workflows_router)
api_router.include_router(admin_router)
api_router.include_router(outreach_router, prefix="/outreach")
