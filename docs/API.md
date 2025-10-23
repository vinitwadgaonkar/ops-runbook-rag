# API Documentation

Complete API reference for the Ops Runbook RAG system.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.your-domain.com`

## Authentication

The API uses JWT-based authentication for production deployments:

```bash
# Get authentication token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username", "password": "your-password"}'

# Use token in requests
curl -H "Authorization: Bearer <token>" "http://localhost:8000/api/v1/query"
```

## Query API

### POST /api/v1/query

Submit a query to the RAG system.

**Request Body**:
```json
{
  "query": "API Gateway is returning 503 errors, what should I do?",
  "context": {
    "service": "api-gateway",
    "severity": "critical",
    "component": "ingress",
    "incident_type": "service_unavailable"
  },
  "max_results": 5,
  "include_sources": true,
  "enable_validation": true
}
```

**Response**:
```json
{
  "answer": "To troubleshoot 503 errors from the API Gateway, follow these steps:\n\n1. Check upstream service health:\n   kubectl get pods -n api-gateway\n   kubectl describe service api-gateway-service\n\n2. Verify load balancer configuration:\n   kubectl logs -n ingress-nginx deployment/ingress-nginx-controller\n\n3. Check resource utilization:\n   kubectl top pods -n api-gateway\n\nIf pods are unhealthy, restart them:\n   kubectl delete pod <unhealthy-pod-name> -n api-gateway\n\nIf resource usage is high, scale up the service:\n   kubectl scale deployment api-gateway --replicas=3 -n api-gateway",
  "sources": [
    {
      "id": "chunk-123",
      "content": "Check upstream service health with kubectl get pods...",
      "document_type": "runbook",
      "metadata": {
        "service": "api-gateway",
        "severity": "critical"
      },
      "relevance_score": 0.95
    }
  ],
  "confidence": 0.92,
  "suggested_actions": [
    "Check service health with kubectl get pods",
    "Verify load balancer configuration",
    "Restart unhealthy pods if needed"
  ],
  "trace_id": "trace-456",
  "latency_ms": 1250,
  "retrieval_metadata": {
    "retrieval_count": 25,
    "reranked_count": 5,
    "avg_relevance_score": 0.87
  }
}
```

**Parameters**:
- `query` (string, required): The question or query text
- `context` (object, optional): Additional context for the query
- `max_results` (integer, optional): Maximum number of results to return (1-20, default: 5)
- `include_sources` (boolean, optional): Whether to include source documents (default: true)
- `enable_validation` (boolean, optional): Whether to validate the response (default: true)

### POST /api/v1/query/batch

Process multiple queries in batch for evaluation.

**Request Body**:
```json
[
  {
    "query": "How to troubleshoot database performance?",
    "context": {"service": "database", "severity": "high"}
  },
  {
    "query": "What are the steps to restart a service?",
    "context": {"service": "user-service", "component": "application"}
  }
]
```

**Response**:
```json
[
  {
    "answer": "Database performance troubleshooting steps...",
    "sources": [...],
    "confidence": 0.88,
    "suggested_actions": [...],
    "trace_id": "trace-789",
    "latency_ms": 1100
  },
  {
    "answer": "Service restart procedures...",
    "sources": [...],
    "confidence": 0.91,
    "suggested_actions": [...],
    "trace_id": "trace-790",
    "latency_ms": 980
  }
]
```

### GET /api/v1/query/suggestions

Get query suggestions based on partial input.

**Parameters**:
- `partial_query` (string, optional): Partial query text
- `context` (object, optional): Context for suggestions
- `limit` (integer, optional): Maximum number of suggestions (default: 10)

**Response**:
```json
[
  "How to troubleshoot 503 errors?",
  "What are the steps to restart a service?",
  "How to check service health?",
  "What commands to run for debugging?",
  "How to rollback a deployment?"
]
```

### GET /api/v1/query/analytics

Get query analytics and performance metrics.

**Parameters**:
- `time_range` (string, optional): Time range for analytics ("1h", "24h", "7d", "30d", default: "24h")

**Response**:
```json
{
  "total_queries": 1250,
  "avg_latency_ms": 1200,
  "success_rate": 0.95,
  "top_queries": [
    {
      "query": "API Gateway 503 errors",
      "count": 45,
      "avg_confidence": 0.89
    }
  ],
  "query_types": {
    "troubleshooting": 60,
    "deployment": 25,
    "monitoring": 15
  },
  "performance_metrics": {
    "p95_latency_ms": 2500,
    "p99_latency_ms": 4000,
    "avg_retrieval_count": 12,
    "avg_confidence_score": 0.87
  }
}
```

## Ingestion API

### POST /api/v1/ingest

Ingest a single document for processing.

**Request** (multipart/form-data):
- `file` (file, required): Document file to ingest
- `document_type` (string, required): Type of document ("runbook", "kb_article", "screenshot", "rca")
- `metadata` (string, optional): JSON metadata for the document
- `force_reprocess` (boolean, optional): Whether to reprocess existing documents

**Response**:
```json
{
  "job_id": "task-123",
  "status": "started",
  "message": "Document ingestion started. Task ID: task-123"
}
```

**Supported File Types**:
- **Runbooks**: `.md`, `.markdown`
- **KB Articles**: `.json`
- **Screenshots**: `.png`, `.jpg`, `.jpeg`
- **RCAs**: `.pdf`

### POST /api/v1/ingest/batch

Batch ingest documents from a directory.

**Request Body**:
```json
{
  "source_directory": "/path/to/documents",
  "document_types": ["runbook", "kb_article"],
  "metadata": {
    "team": "platform",
    "environment": "production"
  }
}
```

**Response**:
```json
{
  "message": "Batch ingestion started for 15 documents",
  "task_ids": ["task-123", "task-124", "task-125"],
  "status": "started"
}
```

### GET /api/v1/ingest/status/{task_id}

Get status of an ingestion task.

**Response**:
```json
{
  "task_id": "task-123",
  "status": "completed",
  "source_path": "/path/to/document.md",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:30:45Z",
  "error_message": null,
  "metadata": {
    "document_id": "doc-456",
    "chunk_count": 8,
    "processing_time_seconds": 40.2
  }
}
```

### GET /api/v1/ingest/stats

Get comprehensive ingestion statistics.

**Response**:
```json
{
  "pipeline_stats": {
    "total_processed": 150,
    "successful": 145,
    "failed": 5,
    "avg_processing_time": 35.2,
    "by_document_type": {
      "runbook": 80,
      "kb_article": 40,
      "screenshot": 20,
      "rca": 10
    }
  },
  "database_stats": {
    "documents": 150,
    "chunks": 1200,
    "queries": 500,
    "evaluation_results": 50
  },
  "vector_stats": {
    "total_embeddings": 1200,
    "by_document_type": {
      "runbook": 800,
      "kb_article": 300,
      "screenshot": 50,
      "rca": 50
    }
  }
}
```

### POST /api/v1/ingest/reprocess/{document_id}

Reprocess an existing document.

**Parameters**:
- `document_id` (string, required): ID of the document to reprocess
- `force` (boolean, optional): Whether to force reprocessing

**Response**:
```json
{
  "message": "Document reprocessing started",
  "task_id": "task-456",
  "document_id": "doc-123",
  "status": "started"
}
```

### DELETE /api/v1/ingest/cleanup

Clean up failed ingestion jobs.

**Parameters**:
- `older_than_hours` (integer, optional): Clean up jobs older than specified hours (default: 24)

**Response**:
```json
{
  "message": "Cleaned up 3 failed jobs",
  "cleaned_count": 3,
  "status": "completed"
}
```

### GET /api/v1/ingest/supported-types

Get supported document types and their specifications.

**Response**:
```json
{
  "supported_types": {
    "runbook": {
      "description": "Markdown runbooks with operational procedures",
      "extensions": [".md", ".markdown"],
      "parser": "MarkdownParser"
    },
    "kb_article": {
      "description": "JSON knowledge base articles",
      "extensions": [".json"],
      "parser": "JSONKBParser"
    },
    "screenshot": {
      "description": "Dashboard screenshots with OCR extraction",
      "extensions": [".png", ".jpg", ".jpeg"],
      "parser": "ScreenshotParser"
    },
    "rca": {
      "description": "PDF root cause analysis documents",
      "extensions": [".pdf"],
      "parser": "PDFRCAParser"
    }
  },
  "max_file_size": "50MB",
  "supported_metadata": [
    "service", "severity", "component", "environment", "team", "tags"
  ]
}
```

## Feedback API

### POST /api/v1/feedback

Submit user feedback for a query response.

**Request Body**:
```json
{
  "query_id": "query-123",
  "feedback_score": 4,
  "feedback_text": "Very helpful, but could use more specific commands",
  "suggested_improvements": [
    "Include more kubectl commands",
    "Add verification steps"
  ]
}
```

**Response**:
```json
{
  "id": "feedback-456",
  "query_id": "query-123",
  "feedback_score": 4,
  "feedback_text": "Very helpful, but could use more specific commands",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### GET /api/v1/feedback/stats

Get feedback statistics for research analysis.

**Parameters**:
- `time_range` (string, optional): Time range for statistics ("1h", "24h", "7d", "30d", default: "7d")

**Response**:
```json
{
  "total_feedback": 250,
  "average_score": 4.2,
  "score_distribution": {
    "score_1": 5,
    "score_2": 10,
    "score_3": 25,
    "score_4": 120,
    "score_5": 90
  },
  "recent_feedback_7d": 45,
  "high_quality_responses": 210,
  "low_quality_responses": 15,
  "quality_rate": 0.84
}
```

### GET /api/v1/feedback/analytics

Get detailed feedback analytics for research insights.

**Response**:
```json
{
  "feedback_trends": [
    {
      "date": "2024-01-14",
      "count": 12,
      "avg_score": 4.1
    },
    {
      "date": "2024-01-15",
      "count": 15,
      "avg_score": 4.3
    }
  ],
  "latency_correlation": {
    "avg_latency_ms": 1200,
    "avg_score": 4.2,
    "correlation": 0.15
  },
  "feedback_by_type": {
    "troubleshooting": 4.3,
    "deployment": 4.1,
    "monitoring": 4.0
  },
  "insights": {
    "total_analyses": 250,
    "trend_period": "7 days",
    "data_quality": "good"
  }
}
```

### GET /api/v1/feedback/queries/{query_id}

Get feedback for a specific query.

**Response**:
```json
{
  "query_id": "query-123",
  "query_text": "API Gateway 503 errors",
  "feedback_score": 4,
  "created_at": "2024-01-15T10:30:00Z",
  "latency_ms": 1250,
  "has_feedback": true
}
```

### GET /api/v1/feedback/export

Export feedback data for research analysis.

**Parameters**:
- `format` (string, optional): Export format ("json", "csv", default: "json")
- `time_range` (string, optional): Time range for export ("1h", "24h", "7d", "30d", default: "30d")

**Response**:
```json
{
  "format": "json",
  "time_range": "30d",
  "record_count": 250,
  "data": [
    {
      "query_id": "query-123",
      "query_text": "API Gateway 503 errors",
      "feedback_score": 4,
      "latency_ms": 1250,
      "created_at": "2024-01-15T10:30:00Z",
      "context": {"service": "api-gateway"}
    }
  ],
  "exported_at": "2024-01-15T10:30:00Z"
}
```

### POST /api/v1/feedback/bulk

Import bulk feedback data for research purposes.

**Request Body**:
```json
[
  {
    "query_id": "query-123",
    "feedback_score": 4
  },
  {
    "query_id": "query-124",
    "feedback_score": 5
  }
]
```

**Response**:
```json
{
  "imported_count": 2,
  "error_count": 0,
  "errors": [],
  "status": "completed"
}
```

## Health API

### GET /api/v1/health

Basic health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "database_healthy": true,
  "redis_healthy": true,
  "vector_extension_available": true
}
```

### GET /api/v1/health/detailed

Detailed health check with system metrics.

**Response**:
```json
{
  "overall_health": {
    "score": 0.95,
    "status": "healthy"
  },
  "components": {
    "database": {
      "healthy": true,
      "score": 1.0,
      "stats": {
        "documents": 150,
        "chunks": 1200,
        "queries": 500
      }
    },
    "vector_extension": {
      "available": true,
      "score": 1.0,
      "stats": {
        "total_embeddings": 1200
      }
    },
    "cache": {
      "healthy": true,
      "score": 0.9,
      "stats": {
        "connected_clients": 5,
        "hit_rate": 0.85
      }
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /api/v1/health/readiness

Kubernetes readiness probe endpoint.

**Response**:
```json
{
  "ready": true,
  "checks": {
    "database": true,
    "vector_extension": true
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /api/v1/health/liveness

Kubernetes liveness probe endpoint.

**Response**:
```json
{
  "alive": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": "2h 30m"
}
```

### GET /api/v1/stats

Get comprehensive system statistics.

**Response**:
```json
{
  "total_documents": 150,
  "total_chunks": 1200,
  "total_queries": 500,
  "total_embeddings": 1200,
  "documents_by_type": {
    "runbook": 80,
    "kb_article": 40,
    "screenshot": 20,
    "rca": 10
  },
  "queries_with_feedback": 250,
  "average_feedback_score": 4.2
}
```

### GET /api/v1/metrics

Get Prometheus-style metrics for monitoring.

**Response**:
```
# HELP database_table_count Total number of records in documents
# TYPE database_table_count counter
database_table_count{table="documents"} 150

# HELP vector_embeddings_total Total number of embeddings
# TYPE vector_embeddings_total counter
vector_embeddings_total 1200

# HELP cache_connected_clients Number of connected cache clients
# TYPE cache_connected_clients gauge
cache_connected_clients 5

# HELP cache_hit_rate Cache hit rate
# TYPE cache_hit_rate gauge
cache_hit_rate 0.85
```

## Error Responses

### Standard Error Format

All API endpoints return errors in the following format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "trace_id": "trace-123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Scenarios

**Invalid Query**:
```json
{
  "error": "Invalid query parameters",
  "detail": "Query text cannot be empty",
  "trace_id": "trace-123"
}
```

**Document Not Found**:
```json
{
  "error": "Document not found",
  "detail": "Document with ID 'doc-123' does not exist",
  "trace_id": "trace-124"
}
```

**Rate Limit Exceeded**:
```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Please try again later.",
  "trace_id": "trace-125"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Query Endpoint**: 100 requests per hour per IP
- **Ingestion Endpoint**: 50 requests per hour per IP
- **Feedback Endpoint**: 200 requests per hour per IP

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## SDK Examples

### Python SDK

```python
import requests

# Query the system
response = requests.post("http://localhost:8000/api/v1/query", json={
    "query": "API Gateway 503 errors",
    "context": {"service": "api-gateway", "severity": "critical"}
})

result = response.json()
print(result["answer"])
```

### JavaScript SDK

```javascript
// Query the system
const response = await fetch("http://localhost:8000/api/v1/query", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        query: "API Gateway 503 errors",
        context: {service: "api-gateway", severity: "critical"}
    })
});

const result = await response.json();
console.log(result.answer);
```

### cURL Examples

```bash
# Query the system
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "API Gateway 503 errors", "context": {"service": "api-gateway"}}'

# Ingest a document
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@runbook.md" \
  -F "document_type=runbook" \
  -F "metadata={\"team\": \"platform\"}"

# Get system health
curl "http://localhost:8000/api/v1/health"
```

## WebSocket Support

For real-time query processing and streaming responses:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/query");

ws.onopen = function() {
    ws.send(JSON.stringify({
        query: "API Gateway 503 errors",
        context: {service: "api-gateway"}
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data.answer);
};
```

## API Versioning

The API uses URL-based versioning:

- **Current Version**: `/api/v1/`
- **Future Versions**: `/api/v2/`, `/api/v3/`, etc.

Version headers are also supported:
```
Accept: application/vnd.ops-runbook.v1+json
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

**For more information, see:**
- [Quick Start Guide](QUICK_START.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Runbook Authoring Guide](RUNBOOK_AUTHORING.md)
