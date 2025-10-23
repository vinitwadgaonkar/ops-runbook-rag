"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from uuid import UUID
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration."""
    RUNBOOK = "runbook"
    KB_ARTICLE = "kb_article"
    SCREENSHOT = "screenshot"
    RCA = "rca"
    INCIDENT_REPORT = "incident_report"


class SeverityLevel(str, Enum):
    """Severity level enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentType(str, Enum):
    """Incident type enumeration."""
    RATE_LIMITING = "rate_limiting"
    CACHE_INVALIDATION = "cache_invalidation"
    DEPLOYMENT_ISSUES = "deployment_issues"
    DATABASE_PROBLEMS = "database_problems"
    SERVICE_MESH = "service_mesh"
    AUTHENTICATION = "authentication"
    NETWORK_CONNECTIVITY = "network_connectivity"


class EvaluationType(str, Enum):
    """Evaluation type enumeration."""
    AUTOMATED = "automated"
    HUMAN = "human"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        from_attributes = True
        use_enum_values = True


# Document schemas
class DocumentBase(BaseSchema):
    """Base document schema."""
    content: str
    metadata: Dict[str, Any]
    document_type: DocumentType
    source_path: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""
    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    id: UUID
    created_at: datetime
    updated_at: datetime


# Chunk schemas
class ChunkBase(BaseSchema):
    """Base chunk schema."""
    content: str
    metadata: Dict[str, Any]
    chunk_index: Optional[int] = None


class ChunkCreate(ChunkBase):
    """Schema for creating a chunk."""
    document_id: UUID


class ChunkResponse(ChunkBase):
    """Schema for chunk response."""
    id: UUID
    document_id: UUID
    created_at: datetime


# Query schemas
class QueryContext(BaseSchema):
    """Query context schema."""
    service: Optional[str] = None
    severity: Optional[SeverityLevel] = None
    component: Optional[str] = None
    environment: Optional[str] = None
    incident_type: Optional[IncidentType] = None


class QueryRequest(BaseSchema):
    """Schema for query request."""
    query: str = Field(..., min_length=1, max_length=1000)
    context: Optional[QueryContext] = None
    max_results: Optional[int] = Field(default=5, ge=1, le=20)
    include_sources: Optional[bool] = Field(default=True)
    enable_validation: Optional[bool] = Field(default=True)


class SourceDocument(BaseSchema):
    """Source document schema."""
    id: UUID
    content: str
    document_type: DocumentType
    metadata: Dict[str, Any]
    relevance_score: float


class QueryResponse(BaseSchema):
    """Schema for query response."""
    answer: str
    sources: List[SourceDocument]
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_actions: List[str]
    trace_id: str
    latency_ms: int
    retrieval_metadata: Dict[str, Any]


# Feedback schemas
class FeedbackRequest(BaseSchema):
    """Schema for feedback request."""
    query_id: UUID
    feedback_score: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None
    suggested_improvements: Optional[List[str]] = None


class FeedbackResponse(BaseSchema):
    """Schema for feedback response."""
    id: UUID
    query_id: UUID
    feedback_score: int
    feedback_text: Optional[str]
    created_at: datetime


# Ingestion schemas
class IngestionRequest(BaseSchema):
    """Schema for ingestion request."""
    source_path: str
    document_type: DocumentType
    metadata: Optional[Dict[str, Any]] = None
    force_reprocess: Optional[bool] = Field(default=False)


class IngestionResponse(BaseSchema):
    """Schema for ingestion response."""
    job_id: UUID
    status: str
    message: str


# Evaluation schemas
class EvaluationRequest(BaseSchema):
    """Schema for evaluation request."""
    query_id: UUID
    metric_name: str
    metric_value: float
    evaluation_type: EvaluationType
    evaluator_id: Optional[str] = None


class EvaluationResponse(BaseSchema):
    """Schema for evaluation response."""
    id: UUID
    query_id: UUID
    metric_name: str
    metric_value: float
    evaluation_type: EvaluationType
    created_at: datetime


# Research experiment schemas
class ExperimentConfig(BaseSchema):
    """Experiment configuration schema."""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]
    evaluation_metrics: List[str]
    incident_types: List[IncidentType]


class ExperimentRequest(BaseSchema):
    """Schema for experiment request."""
    config: ExperimentConfig
    dataset_path: Optional[str] = None


class ExperimentResponse(BaseSchema):
    """Schema for experiment response."""
    id: UUID
    name: str
    status: str
    created_at: datetime
    results: Optional[Dict[str, Any]] = None


# Health check schemas
class HealthCheck(BaseSchema):
    """Health check schema."""
    status: str
    timestamp: datetime
    version: str
    database_healthy: bool
    redis_healthy: bool
    vector_extension_available: bool


class SystemStats(BaseSchema):
    """System statistics schema."""
    total_documents: int
    total_chunks: int
    total_queries: int
    total_embeddings: int
    documents_by_type: Dict[str, int]
    queries_with_feedback: int
    average_feedback_score: Optional[float] = None


# Validation schemas
class ValidationResult(BaseSchema):
    """Validation result schema."""
    is_valid: bool
    confidence: float
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]


class CommandValidation(BaseSchema):
    """Command validation schema."""
    command: str
    is_safe: bool
    syntax_valid: bool
    service_aware: bool
    confidence: float
    warnings: List[str]


# Research analytics schemas
class RetrievalAnalytics(BaseSchema):
    """Retrieval analytics schema."""
    date: datetime
    total_queries: int
    avg_latency_ms: float
    avg_feedback_score: Optional[float]
    high_rating_count: int


class ModelPerformance(BaseSchema):
    """Model performance schema."""
    model_name: str
    accuracy: float
    latency_ms: float
    token_usage: int
    cost_usd: float
    evaluation_date: datetime


# Error schemas
class ErrorResponse(BaseSchema):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: datetime


# Utility functions
def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean metadata."""
    # Remove None values
    cleaned = {k: v for k, v in metadata.items() if v is not None}
    
    # Ensure required fields have valid values
    if "severity" in cleaned:
        if cleaned["severity"] not in [e.value for e in SeverityLevel]:
            cleaned["severity"] = "medium"
    
    return cleaned


def validate_query_context(context: Optional[QueryContext]) -> Optional[QueryContext]:
    """Validate query context."""
    if context is None:
        return None
    
    # Validate severity level
    if context.severity and context.severity not in [e.value for e in SeverityLevel]:
        context.severity = None
    
    # Validate incident type
    if context.incident_type and context.incident_type not in [e.value for e in IncidentType]:
        context.incident_type = None
    
    return context
