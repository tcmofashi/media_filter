"""FastAPI application with health check endpoint"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.error_handler import register_error_handlers
from src.services.pipeline_jobs import pipeline_job_manager
from src.logger import get_logger, setup_file_logging
from src.storage.database import db

# Initialize file logging on startup
log_file = setup_file_logging()
logger = get_logger(__name__)
logger.info(f"File logging initialized: {log_file}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources"""
    # Startup: Initialize database
    await db.init()
    logger.info("Database initialized successfully")
    
    # Startup: Initialize task queue
    from src.services.task_queue import task_queue
    await task_queue.start()
    logger.info("Task queue started successfully")
    
    yield
    
    # Shutdown: Stop task queue
    await task_queue.stop()
    logger.info("Task queue stopped")

    # Shutdown: Stop background pipeline jobs
    await pipeline_job_manager.shutdown()
    logger.info("Pipeline job manager stopped")
    
    # Shutdown: Close database connection
    await db.close()
    logger.info("Database connection closed")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="MediaFlusher API",
        description="Frozen CLIP media scoring and Telegram curation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from src.api.routes import api_router

    app.include_router(api_router)

    # Register error handlers
    register_error_handlers(app)

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy"}

    logger.info("FastAPI application created")

    return app


app = create_app()
