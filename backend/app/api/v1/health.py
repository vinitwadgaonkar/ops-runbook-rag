"""
Health check API endpoint for system monitoring.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from ...core.database import get_db, check_db_connection, get_vector_extension_status
from ...core.cache import get_cache_stats
from ...models.schemas import HealthCheck, SystemStats

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health_check(
    db: Session = Depends(get_db)
) -> HealthCheck:
    """
    Comprehensive health check for the research system.
    
    Checks:
    - Database connectivity
    - Vector extension availability
    - Cache system health
    - System performance metrics
    """
    try:
        # Check database health
        db_healthy = check_db_connection()
        vector_available = get_vector_extension_status()
        
        # Check cache health
        cache_stats = get_cache_stats()
        cache_healthy = cache_stats["status"] == "connected"
        
        # Determine overall status
        if db_healthy and cache_healthy and vector_available:
            status = "healthy"
        elif db_healthy and cache_healthy:
            status = "degraded"  # Vector extension missing
        else:
            status = "unhealthy"
        
        return HealthCheck(
            status=status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_healthy=db_healthy,
            redis_healthy=cache_healthy,
            vector_extension_available=vector_available
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_healthy=False,
            redis_healthy=False,
            vector_extension_available=False
        )


@router.get("/health/detailed")
async def detailed_health_check(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Detailed health check with system metrics.
    """
    try:
        from ...core.database import db_manager
        
        # Basic health checks
        db_healthy = check_db_connection()
        vector_available = get_vector_extension_status()
        cache_stats = get_cache_stats()
        
        # Get system statistics
        table_stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
        
        # Calculate health scores
        db_score = 1.0 if db_healthy else 0.0
        vector_score = 1.0 if vector_available else 0.0
        cache_score = 1.0 if cache_stats["status"] == "connected" else 0.0
        
        overall_score = (db_score + vector_score + cache_score) / 3.0
        
        return {
            "overall_health": {
                "score": overall_score,
                "status": "healthy" if overall_score >= 0.8 else "degraded" if overall_score >= 0.5 else "unhealthy"
            },
            "components": {
                "database": {
                    "healthy": db_healthy,
                    "score": db_score,
                    "stats": table_stats
                },
                "vector_extension": {
                    "available": vector_available,
                    "score": vector_score,
                    "stats": vector_stats
                },
                "cache": {
                    "healthy": cache_stats["status"] == "connected",
                    "score": cache_score,
                    "stats": cache_stats
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/readiness")
async def readiness_check(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    """
    try:
        # Check critical dependencies
        db_healthy = check_db_connection()
        vector_available = get_vector_extension_status()
        
        # System is ready if database and vector extension are available
        ready = db_healthy and vector_available
        
        return {
            "ready": ready,
            "checks": {
                "database": db_healthy,
                "vector_extension": vector_available
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "ready": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/health/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe endpoint.
    """
    try:
        # Simple liveness check - just verify the service is responding
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "unknown"  # Would need to track actual uptime
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return {
            "alive": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/stats", response_model=SystemStats)
async def system_stats(
    db: Session = Depends(get_db)
) -> SystemStats:
    """
    Get comprehensive system statistics.
    """
    try:
        from ...core.database import db_manager
        
        # Get database statistics
        table_stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
        
        # Get cache statistics
        cache_stats = get_cache_stats()
        
        # Calculate feedback statistics
        from sqlalchemy import func
        from ...models.database import Query
        
        queries_with_feedback = db.query(Query).filter(Query.feedback_score.isnot(None)).count()
        avg_feedback_score = db.query(func.avg(Query.feedback_score)).filter(
            Query.feedback_score.isnot(None)
        ).scalar()
        
        return SystemStats(
            total_documents=table_stats.get("documents", 0),
            total_chunks=table_stats.get("chunks", 0),
            total_queries=table_stats.get("queries", 0),
            total_embeddings=vector_stats.get("total_embeddings", 0),
            documents_by_type=vector_stats.get("by_document_type", {}),
            queries_with_feedback=queries_with_feedback,
            average_feedback_score=float(avg_feedback_score) if avg_feedback_score else None
        )
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get Prometheus-style metrics for monitoring.
    """
    try:
        from ...core.database import db_manager
        
        # Get system statistics
        table_stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
        cache_stats = get_cache_stats()
        
        # Format as Prometheus metrics
        metrics = []
        
        # Database metrics
        for table, count in table_stats.items():
            metrics.append(f"# HELP database_table_count Total number of records in {table}")
            metrics.append(f"# TYPE database_table_count counter")
            metrics.append(f"database_table_count{{table=\"{table}\"}} {count}")
        
        # Vector metrics
        metrics.append(f"# HELP vector_embeddings_total Total number of embeddings")
        metrics.append(f"# TYPE vector_embeddings_total counter")
        metrics.append(f"vector_embeddings_total {vector_stats.get('total_embeddings', 0)}")
        
        # Cache metrics
        if cache_stats["status"] == "connected":
            cache_info = cache_stats.get("stats", {})
            metrics.append(f"# HELP cache_connected_clients Number of connected cache clients")
            metrics.append(f"# TYPE cache_connected_clients gauge")
            metrics.append(f"cache_connected_clients {cache_info.get('connected_clients', 0)}")
            
            metrics.append(f"# HELP cache_hit_rate Cache hit rate")
            metrics.append(f"# TYPE cache_hit_rate gauge")
            metrics.append(f"cache_hit_rate {cache_info.get('hit_rate', 0)}")
        
        return {
            "metrics": "\n".join(metrics),
            "format": "prometheus",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
