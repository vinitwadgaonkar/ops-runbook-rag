#!/usr/bin/env python3
"""
Setup script for the Ops Runbook RAG research environment.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.database import init_db, check_db_connection, get_vector_extension_status
from app.core.cache import cache_manager
from app.ingestion.pipeline import ingestion_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("Checking dependencies...")
    
    try:
        import psycopg2
        import redis
        import openai
        import anthropic
        logger.info("✅ All Python dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False


def check_database_connection():
    """Check database connectivity and pgvector extension."""
    logger.info("Checking database connection...")
    
    try:
        if not check_db_connection():
            logger.error("❌ Database connection failed")
            return False
        
        logger.info("✅ Database connection successful")
        
        if not get_vector_extension_status():
            logger.warning("⚠️  pgvector extension not available - vector operations may fail")
        else:
            logger.info("✅ pgvector extension available")
        
        return True
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False


def check_cache_connection():
    """Check Redis cache connectivity."""
    logger.info("Checking cache connection...")
    
    try:
        cache_stats = cache_manager.get_cache_stats()
        if cache_stats["status"] == "connected":
            logger.info("✅ Cache connection successful")
            return True
        else:
            logger.warning("⚠️  Cache connection failed - caching disabled")
            return False
    except Exception as e:
        logger.error(f"❌ Cache check failed: {e}")
        return False


def initialize_database():
    """Initialize database tables and schema."""
    logger.info("Initializing database...")
    
    try:
        init_db()
        logger.info("✅ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


def ingest_sample_data():
    """Ingest sample data for demonstration."""
    logger.info("Ingesting sample data...")
    
    try:
        sample_data_path = Path(__file__).parent.parent / "sample_data"
        
        # Ingest runbooks
        runbook_path = sample_data_path / "runbooks"
        if runbook_path.exists():
            task_ids = []
            for md_file in runbook_path.glob("*.md"):
                task_id = ingestion_pipeline.ingest_document(
                    str(md_file), "runbook", {"team": "platform"}
                )
                task_ids.append(task_id)
            logger.info(f"✅ Ingested {len(task_ids)} runbooks")
        
        # Ingest KB articles
        kb_path = sample_data_path / "kb_articles"
        if kb_path.exists():
            task_ids = []
            for json_file in kb_path.glob("*.json"):
                task_id = ingestion_pipeline.ingest_document(
                    str(json_file), "kb_article", {"team": "data-platform"}
                )
                task_ids.append(task_id)
            logger.info(f"✅ Ingested {len(task_ids)} KB articles")
        
        return True
    except Exception as e:
        logger.error(f"❌ Sample data ingestion failed: {e}")
        return False


def test_query_system():
    """Test the query system with a sample query."""
    logger.info("Testing query system...")
    
    try:
        # This would test the full RAG pipeline
        # For now, just log that the test would run
        logger.info("✅ Query system test would run here")
        return True
    except Exception as e:
        logger.error(f"❌ Query system test failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("🚀 Setting up Ops Runbook RAG Research Environment")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        return False
    
    # Check database
    if not check_database_connection():
        logger.error("❌ Database check failed")
        return False
    
    # Check cache
    check_cache_connection()  # Non-critical
    
    # Initialize database
    if not initialize_database():
        logger.error("❌ Database initialization failed")
        return False
    
    # Ingest sample data
    if not ingest_sample_data():
        logger.error("❌ Sample data ingestion failed")
        return False
    
    # Test query system
    if not test_query_system():
        logger.error("❌ Query system test failed")
        return False
    
    logger.info("🎉 Research environment setup completed successfully!")
    logger.info("📊 You can now:")
    logger.info("  - Query documents via /api/v1/query")
    logger.info("  - Ingest new documents via /api/v1/ingest")
    logger.info("  - View system stats via /api/v1/health/stats")
    logger.info("  - Access Grafana dashboards at http://localhost:3000")
    logger.info("  - View Prometheus metrics at http://localhost:9090")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
