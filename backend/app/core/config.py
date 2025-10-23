"""
Research configuration settings for the Ops Runbook RAG system.
"""

from typing import Optional, List
from pydantic import BaseSettings, Field
import os


class Settings(BaseSettings):
    """Research environment configuration."""
    
    # API Configuration
    app_name: str = "Ops Runbook RAG Research"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://research:research@localhost:5432/ops_runbook_rag",
        env="DATABASE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")  # 1 hour
    
    # Message Queue Configuration
    rabbitmq_url: str = Field(
        default="amqp://research:research@localhost:5672/",
        env="RABBITMQ_URL"
    )
    
    # LLM Provider Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    
    # Model Configuration
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=3072, env="EMBEDDING_DIMENSIONS")
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # Retrieval Configuration
    top_k_initial: int = Field(default=50, env="TOP_K_INITIAL")
    top_k_final: int = Field(default=10, env="TOP_K_FINAL")
    top_k_llm: int = Field(default=5, env="TOP_K_LLM")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Temporal Decay Configuration
    temporal_decay_lambda: float = Field(default=0.05, env="TEMPORAL_DECAY_LAMBDA")
    
    # Reranking Configuration
    rerank_model: str = Field(default="cohere", env="RERANK_MODEL")
    rerank_top_k: int = Field(default=10, env="RERANK_TOP_K")
    
    # Validation Configuration
    enable_command_validation: bool = Field(default=True, env="ENABLE_COMMAND_VALIDATION")
    enable_syntax_validation: bool = Field(default=True, env="ENABLE_SYNTAX_VALIDATION")
    enable_hallucination_detection: bool = Field(default=True, env="ENABLE_HALLUCINATION_DETECTION")
    
    # Observability Configuration
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    
    # Research Configuration
    evaluation_dataset_path: str = Field(
        default="/app/sample_data/evaluation/incident_set.json",
        env="EVALUATION_DATASET_PATH"
    )
    sample_data_path: str = Field(default="/app/sample_data", env="SAMPLE_DATA_PATH")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # Security
    secret_key: str = Field(default="research-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Research-specific configuration
class ResearchConfig:
    """Research experiment configuration."""
    
    # Experiment tracking
    experiment_name: str = "ops_runbook_rag_v1"
    experiment_version: str = "1.0.0"
    
    # Evaluation metrics
    evaluation_metrics: List[str] = [
        "bleu_score",
        "rouge_l",
        "semantic_similarity",
        "actionability_score",
        "response_time",
        "retrieval_accuracy"
    ]
    
    # Incident types for evaluation
    incident_types: List[str] = [
        "rate_limiting",
        "cache_invalidation", 
        "deployment_issues",
        "database_problems",
        "service_mesh",
        "authentication",
        "network_connectivity"
    ]
    
    # Document types
    document_types: List[str] = [
        "runbook",
        "kb_article", 
        "screenshot",
        "rca",
        "incident_report"
    ]
    
    # Service metadata fields
    service_metadata_fields: List[str] = [
        "service",
        "severity",
        "component",
        "environment",
        "team",
        "tags"
    ]


# Research configuration instance
research_config = ResearchConfig()
