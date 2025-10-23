"""
SQLAlchemy database models for the research system.
"""

from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ARRAY, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ..core.database import Base


class Document(Base):
    """Document model for storing ingested content."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, nullable=False)
    document_type = Column(String(50), nullable=False)
    source_path = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, type={self.document_type})>"


class Chunk(Base):
    """Chunk model for storing document segments with embeddings."""
    
    __tablename__ = "chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column("embedding", String)  # Will be converted to vector type by pgvector
    metadata = Column(JSON, nullable=False)
    chunk_index = Column(Integer)
    ts_vector = Column("ts_vector", String)  # Will be converted to tsvector type
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


class Query(Base):
    """Query model for storing user queries and responses."""
    
    __tablename__ = "queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    context = Column(JSON)
    retrieved_chunks = Column(ARRAY(UUID(as_uuid=True)))
    llm_response = Column(Text)
    latency_ms = Column(Integer)
    feedback_score = Column(Integer)  # 1-5 rating
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    evaluation_results = relationship("EvaluationResult", back_populates="query")
    
    def __repr__(self):
        return f"<Query(id={self.id}, score={self.feedback_score})>"


class EvaluationResult(Base):
    """Evaluation result model for storing research metrics."""
    
    __tablename__ = "evaluation_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    evaluation_type = Column(String(50), nullable=False)  # 'automated' or 'human'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    query = relationship("Query", back_populates="evaluation_results")
    
    def __repr__(self):
        return f"<EvaluationResult(metric={self.metric_name}, value={self.metric_value})>"


class IngestionJob(Base):
    """Ingestion job model for tracking document processing."""
    
    __tablename__ = "ingestion_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    source_path = Column(Text)
    metadata = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<IngestionJob(id={self.id}, type={self.job_type}, status={self.status})>"


# Research-specific models for experiment tracking
class Experiment(Base):
    """Experiment model for tracking research experiments."""
    
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    config = Column(JSON, nullable=False)
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<Experiment(id={self.id}, name={self.name}, status={self.status})>"


class ExperimentRun(Base):
    """Experiment run model for tracking individual experiment executions."""
    
    __tablename__ = "experiment_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    run_name = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=False)
    results = Column(JSON)
    status = Column(String(20), nullable=False, default="running")
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    def __repr__(self):
        return f"<ExperimentRun(id={self.id}, name={self.run_name}, status={self.status})>"


# Utility functions for model operations
def get_document_by_type(session, document_type: str, limit: int = 100):
    """Get documents by type for research analysis."""
    return session.query(Document).filter(
        Document.document_type == document_type
    ).limit(limit).all()


def get_chunks_by_metadata(session, metadata_filter: dict, limit: int = 100):
    """Get chunks filtered by metadata for research analysis."""
    query = session.query(Chunk)
    
    for key, value in metadata_filter.items():
        query = query.filter(Chunk.metadata[key].astext == str(value))
    
    return query.limit(limit).all()


def get_queries_with_feedback(session, min_score: int = 1, limit: int = 100):
    """Get queries with feedback scores for analysis."""
    return session.query(Query).filter(
        Query.feedback_score >= min_score
    ).limit(limit).all()


def get_evaluation_metrics(session, metric_name: str = None, limit: int = 1000):
    """Get evaluation metrics for research analysis."""
    query = session.query(EvaluationResult)
    
    if metric_name:
        query = query.filter(EvaluationResult.metric_name == metric_name)
    
    return query.limit(limit).all()
