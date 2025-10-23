"""
FastAPI application entry point for the Ops Runbook RAG research system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

from .core.config import settings
from .core.database import init_db, check_db_connection, get_vector_extension_status
from .core.observability import (
    observability_manager, 
    instrument_fastapi, 
    instrument_sqlalchemy,
    instrument_redis
)
from .api.v1 import query, ingest, feedback, health
from .models.schemas import HealthCheck, SystemStats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Ops Runbook RAG Research System")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Check database health
        if not check_db_connection():
            raise Exception("Database connection failed")
        logger.info("Database connection verified")
        
        # Check pgvector extension
        if not get_vector_extension_status():
            logger.warning("pgvector extension not available - vector operations may fail")
        else:
            logger.info("pgvector extension verified")
        
        # Setup instrumentation
        instrument_sqlalchemy(app.state.engine)
        instrument_redis()
        logger.info("Observability instrumentation enabled")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ops Runbook RAG Research System")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Research system for retrieval-augmented generation in operational knowledge management",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Instrument FastAPI
instrument_fastapi(app)

# Include API routers
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(ingest.router, prefix="/api/v1", tags=["ingestion"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with system information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs_url": "/docs" if settings.debug else None
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Comprehensive health check endpoint."""
    from .core.database import db_manager
    from .core.cache import get_cache_stats
    
    # Check database health
    db_healthy = check_db_connection()
    vector_available = get_vector_extension_status()
    
    # Check cache health
    cache_stats = get_cache_stats()
    cache_healthy = cache_stats["status"] == "connected"
    
    # Get system stats
    try:
        stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        stats = {}
        vector_stats = {}
    
    return HealthCheck(
        status="healthy" if all([db_healthy, cache_healthy]) else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        database_healthy=db_healthy,
        redis_healthy=cache_healthy,
        vector_extension_available=vector_available
    )


@app.get("/stats", response_model=SystemStats)
async def system_stats():
    """Get system statistics for research monitoring."""
    from .core.database import db_manager
    from .core.cache import get_cache_stats
    
    try:
        # Get database stats
        table_stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
        
        # Get cache stats
        cache_stats = get_cache_stats()
        
        # Calculate average feedback score
        from .core.database import SessionLocal
        from .models.database import Query
        
        with SessionLocal() as db:
            feedback_queries = db.query(Query).filter(Query.feedback_score.isnot(None)).all()
            avg_feedback = sum(q.feedback_score for q in feedback_queries) / len(feedback_queries) if feedback_queries else None
        
        return SystemStats(
            total_documents=table_stats.get("documents", 0),
            total_chunks=table_stats.get("chunks", 0),
            total_queries=table_stats.get("queries", 0),
            total_embeddings=vector_stats.get("total_embeddings", 0),
            documents_by_type=vector_stats.get("by_document_type", {}),
            queries_with_feedback=len(feedback_queries),
            average_feedback_score=avg_feedback
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")


@app.get("/research/metrics")
async def research_metrics():
    """Get research-specific metrics for experiment tracking."""
    from .core.database import db_manager
    
    try:
        # Get comprehensive stats
        table_stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
        
        # Get recent query performance
        from .core.database import SessionLocal
        from .models.database import Query
        from sqlalchemy import func
        
        with SessionLocal() as db:
            # Recent query stats
            recent_queries = db.query(Query).filter(
                Query.created_at >= func.now() - func.interval('24 hours')
            ).all()
            
            avg_latency = sum(q.latency_ms for q in recent_queries if q.latency_ms) / len(recent_queries) if recent_queries else 0
            
            # Feedback distribution
            feedback_dist = {}
            for score in range(1, 6):
                count = db.query(Query).filter(Query.feedback_score == score).count()
                feedback_dist[f"score_{score}"] = count
        
        return {
            "data_volume": {
                "documents": table_stats.get("documents", 0),
                "chunks": table_stats.get("chunks", 0),
                "embeddings": vector_stats.get("total_embeddings", 0),
                "queries": table_stats.get("queries", 0)
            },
            "performance": {
                "avg_latency_ms": avg_latency,
                "recent_queries_24h": len(recent_queries)
            },
            "feedback_distribution": feedback_dist,
            "document_types": vector_stats.get("by_document_type", {}),
            "system_health": {
                "database": check_db_connection(),
                "vector_extension": get_vector_extension_status(),
                "cache": get_cache_stats()["status"] == "connected"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get research metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve research metrics")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
