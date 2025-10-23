"""
Database configuration and connection management for the research system.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=settings.debug,
    connect_args={
        "options": "-c timezone=utc"
    }
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

# Metadata for schema management
metadata = MetaData()


def get_db():
    """
    Dependency to get database session.
    Used with FastAPI dependency injection.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.
    Called during application startup.
    """
    try:
        # Import all models to ensure they are registered
        from ..models.database import Document, Chunk, Query, EvaluationResult, IngestionJob
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def check_db_connection():
    """
    Check database connection health.
    Returns True if connection is healthy.
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_vector_extension_status():
    """
    Check if pgvector extension is available.
    Returns True if pgvector is installed and working.
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )
            return result.fetchone() is not None
    except Exception as e:
        logger.error(f"Failed to check pgvector extension: {e}")
        return False


class DatabaseManager:
    """
    Database management utilities for research operations.
    """
    
    def __init__(self):
        self.engine = engine
        self.session_factory = SessionLocal
    
    def create_session(self):
        """Create a new database session."""
        return self.session_factory()
    
    def execute_query(self, query, params=None):
        """Execute a raw SQL query."""
        with self.engine.connect() as connection:
            result = connection.execute(query, params or {})
            return result.fetchall()
    
    def get_table_stats(self):
        """Get statistics about database tables."""
        stats = {}
        
        tables = ['documents', 'chunks', 'queries', 'evaluation_results', 'ingestion_jobs']
        
        for table in tables:
            try:
                with self.engine.connect() as connection:
                    result = connection.execute(f"SELECT COUNT(*) FROM {table}")
                    count = result.fetchone()[0]
                    stats[table] = count
            except Exception as e:
                logger.error(f"Failed to get stats for table {table}: {e}")
                stats[table] = 0
        
        return stats
    
    def get_vector_stats(self):
        """Get statistics about vector embeddings."""
        try:
            with self.engine.connect() as connection:
                # Count total embeddings
                result = connection.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
                total_embeddings = result.fetchone()[0]
                
                # Count embeddings by document type
                result = connection.execute("""
                    SELECT d.document_type, COUNT(c.id) 
                    FROM documents d 
                    JOIN chunks c ON d.id = c.document_id 
                    WHERE c.embedding IS NOT NULL 
                    GROUP BY d.document_type
                """)
                by_type = dict(result.fetchall())
                
                return {
                    "total_embeddings": total_embeddings,
                    "by_document_type": by_type
                }
        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            return {"total_embeddings": 0, "by_document_type": {}}
    
    def cleanup_old_data(self, days_old=30):
        """Clean up old evaluation data and queries."""
        try:
            with self.engine.connect() as connection:
                # Delete old evaluation results
                result = connection.execute("""
                    DELETE FROM evaluation_results 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                """, (days_old,))
                
                deleted_evaluations = result.rowcount
                
                # Delete old queries (keep recent ones for analysis)
                result = connection.execute("""
                    DELETE FROM queries 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                    AND feedback_score IS NULL
                """, (days_old,))
                
                deleted_queries = result.rowcount
                
                logger.info(f"Cleaned up {deleted_evaluations} evaluation results and {deleted_queries} queries")
                return {"evaluations_deleted": deleted_evaluations, "queries_deleted": deleted_queries}
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {"evaluations_deleted": 0, "queries_deleted": 0}


# Global database manager instance
db_manager = DatabaseManager()
