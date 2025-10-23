"""
OpenTelemetry observability setup for research instrumentation.
"""

import logging
from typing import Optional, Dict, Any
from opentelemetry import trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

from .config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics
query_counter = Counter('rag_queries_total', 'Total number of queries', ['status'])
query_latency = Histogram('rag_query_latency_seconds', 'Query latency in seconds', ['stage'])
embedding_latency = Histogram('rag_embedding_latency_seconds', 'Embedding generation latency', ['model'])
llm_latency = Histogram('rag_llm_latency_seconds', 'LLM response latency', ['provider', 'model'])
llm_tokens = Counter('rag_llm_tokens_total', 'Total LLM tokens used', ['provider', 'model', 'type'])
cache_hits = Counter('rag_cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses = Counter('rag_cache_misses_total', 'Cache misses', ['cache_type'])
retrieval_accuracy = Gauge('rag_retrieval_accuracy', 'Retrieval accuracy score')
feedback_score = Histogram('rag_feedback_score', 'User feedback scores', buckets=[1, 2, 3, 4, 5])


class ObservabilityManager:
    """
    Centralized observability management for research instrumentation.
    """
    
    def __init__(self):
        self.tracer = None
        self.meter = None
        self._setup_tracing()
        self._setup_metrics()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        if not settings.enable_tracing:
            logger.info("Tracing disabled")
            return
        
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": "ops-runbook-rag",
                "service.version": "1.0.0",
                "service.namespace": "research"
            })
            
            # Setup tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))
            self.tracer = trace.get_tracer(__name__)
            
            # Setup span processor
            if settings.jaeger_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=settings.jaeger_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            
            logger.info("Tracing setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup tracing: {e}")
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics."""
        if not settings.enable_metrics:
            logger.info("Metrics disabled")
            return
        
        try:
            # Create resource
            resource = Resource.create({
                "service.name": "ops-runbook-rag",
                "service.version": "1.0.0"
            })
            
            # Setup meter provider
            meter_provider = MeterProvider(resource=resource)
            
            # Start Prometheus metrics server
            start_http_server(settings.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {settings.prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to setup metrics: {e}")
    
    def get_tracer(self):
        """Get the tracer instance."""
        return self.tracer
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a new span with attributes."""
        if not self.tracer:
            return None
        
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span


class ResearchInstrumentation:
    """
    Research-specific instrumentation for tracking experiments and metrics.
    """
    
    def __init__(self, observability_manager: ObservabilityManager):
        self.obs_manager = observability_manager
        self.tracer = observability_manager.get_tracer()
    
    def track_query(self, query_text: str, context: Optional[Dict[str, Any]] = None):
        """Track a query with full instrumentation."""
        with self.tracer.start_span("query.receive") as span:
            span.set_attribute("query.text", query_text)
            if context:
                for key, value in context.items():
                    span.set_attribute(f"query.context.{key}", str(value))
            
            # Increment query counter
            query_counter.labels(status="received").inc()
            
            return span
    
    def track_embedding_generation(self, text: str, model: str):
        """Track embedding generation."""
        start_time = time.time()
        
        with self.tracer.start_span("retrieval.embed") as span:
            span.set_attribute("embedding.model", model)
            span.set_attribute("embedding.text_length", len(text))
            
            def finish_embedding():
                latency = time.time() - start_time
                embedding_latency.labels(model=model).observe(latency)
                span.set_attribute("embedding.latency_ms", latency * 1000)
            
            return finish_embedding
    
    def track_vector_search(self, query_vector: list, top_k: int):
        """Track vector search operation."""
        with self.tracer.start_span("retrieval.vector_search") as span:
            span.set_attribute("search.top_k", top_k)
            span.set_attribute("search.vector_dim", len(query_vector))
            
            return span
    
    def track_reranking(self, candidates: int, model: str):
        """Track reranking operation."""
        with self.tracer.start_span("retrieval.rerank") as span:
            span.set_attribute("rerank.candidates", candidates)
            span.set_attribute("rerank.model", model)
            
            return span
    
    def track_llm_call(self, provider: str, model: str, prompt_tokens: int):
        """Track LLM API call."""
        start_time = time.time()
        
        with self.tracer.start_span("generation.llm_call") as span:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            
            def finish_llm_call(completion_tokens: int, total_tokens: int):
                latency = time.time() - start_time
                llm_latency.labels(provider=provider, model=model).observe(latency)
                llm_tokens.labels(provider=provider, model=model, type="prompt").inc(prompt_tokens)
                llm_tokens.labels(provider=provider, model=model, type="completion").inc(completion_tokens)
                span.set_attribute("llm.completion_tokens", completion_tokens)
                span.set_attribute("llm.total_tokens", total_tokens)
                span.set_attribute("llm.latency_ms", latency * 1000)
            
            return finish_llm_call
    
    def track_validation(self, validation_type: str, is_valid: bool):
        """Track response validation."""
        with self.tracer.start_span("validation.check") as span:
            span.set_attribute("validation.type", validation_type)
            span.set_attribute("validation.is_valid", is_valid)
            
            return span
    
    def track_cache_operation(self, operation: str, cache_type: str, hit: bool):
        """Track cache operations."""
        if hit:
            cache_hits.labels(cache_type=cache_type).inc()
        else:
            cache_misses.labels(cache_type=cache_type).inc()
    
    def track_feedback(self, score: int, query_id: str):
        """Track user feedback."""
        feedback_score.observe(score)
        
        with self.tracer.start_span("feedback.record") as span:
            span.set_attribute("feedback.score", score)
            span.set_attribute("feedback.query_id", query_id)
            
            return span
    
    def track_experiment(self, experiment_id: str, metric_name: str, value: float):
        """Track experiment metrics."""
        with self.tracer.start_span("experiment.metric") as span:
            span.set_attribute("experiment.id", experiment_id)
            span.set_attribute("experiment.metric", metric_name)
            span.set_attribute("experiment.value", value)
            
            return span


# Global observability instances
observability_manager = ObservabilityManager()
research_instrumentation = ResearchInstrumentation(observability_manager)


def instrument_fastapi(app):
    """Instrument FastAPI application."""
    if settings.enable_tracing:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")


def instrument_sqlalchemy(engine):
    """Instrument SQLAlchemy engine."""
    if settings.enable_tracing:
        SQLAlchemyInstrumentor().instrument(engine=engine)
        logger.info("SQLAlchemy instrumentation enabled")


def instrument_redis():
    """Instrument Redis client."""
    if settings.enable_tracing:
        RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled")


def get_sli_slo_metrics() -> Dict[str, Any]:
    """Get SLI/SLO metrics for monitoring."""
    return {
        "latency_p95": "< 3s",
        "latency_p99": "< 5s", 
        "availability": "> 99.5%",
        "accuracy": "> 75%",
        "token_efficiency": "< 5000 tokens/query"
    }


def track_query_latency(stage: str):
    """Decorator to track query latency by stage."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                query_latency.labels(stage=stage).observe(latency)
        return wrapper
    return decorator
