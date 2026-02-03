"""
SPIKE API - Main Application Entry Point
AI Wealth Intelligence Platform API
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug: {settings.DEBUG}")

    # TODO: Initialize Redis connection pool
    # TODO: Initialize market data connections
    # TODO: Start background tasks (FinScore updates, etc.)

    yield

    # Shutdown
    print("Shutting down...")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    # SPIKE - AI Wealth Intelligence Platform API

    India's first AI-powered wealth intelligence platform providing:

    - **FinScore**: Universal 0-10 stock ratings combining 9 dimensions
    - **Legend Agents**: AI trained on Buffett, Lynch, Dalio philosophies
    - **Strategy-GPT**: Natural language to quantitative strategies
    - **Portfolio Autopilot**: Autonomous portfolio management
    - **Smart Themes**: AI-curated investment baskets

    ## SEBI Compliance
    This API is built with full SEBI RA/IA compliance. All data is localized in India.

    ## Rate Limits
    - Free tier: 100 requests/minute
    - Pro tier: 1000 requests/minute
    - Enterprise: Unlimited
    """,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_V1_PREFIX}/docs",
        "health": "/health",
    }


# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)
