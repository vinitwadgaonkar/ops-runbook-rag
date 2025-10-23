"""
Query API endpoint for the RAG system.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
import time
import uuid

from ...core.database import get_db
from ...core.observability import research_instrumentation
from ...models.schemas import QueryRequest, QueryResponse, SourceDocument
from ...retrieval.vector_store import VectorStore
from ...retrieval.hybrid_search import HybridSearch
from ...retrieval.reranker import Reranker
from ...generation.llm_client import LLMClient
from ...generation.validators import ResponseValidator

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
vector_store = VectorStore()
hybrid_search = HybridSearch()
reranker = Reranker()
llm_client = LLMClient()
response_validator = ResponseValidator()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> QueryResponse:
    """
    Query documents using the full RAG pipeline.
    
    This endpoint processes queries through:
    1. Query embedding generation
    2. Hybrid retrieval (dense + sparse)
    3. Reranking with temporal decay
    4. LLM generation with incident-aware prompting
    5. Response validation
    """
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    
    try:
        # Track query start
        research_instrumentation.track_query(request.query, request.context)
        
        # Step 1: Generate query embedding
        query_embedding = await vector_store.embed_query(request.query)
        
        # Step 2: Hybrid retrieval
        retrieval_results = await hybrid_search.search(
            query=request.query,
            query_embedding=query_embedding,
            context=request.context,
            top_k=request.max_results * 2  # Get more for reranking
        )
        
        # Step 3: Rerank results
        reranked_results = await reranker.rerank(
            query=request.query,
            documents=retrieval_results,
            top_k=request.max_results
        )
        
        # Step 4: Generate LLM response
        llm_response = await llm_client.generate_response(
            query=request.query,
            context=request.context,
            retrieved_documents=reranked_results,
            incident_type=request.context.get("incident_type") if request.context else None
        )
        
        # Step 5: Validate response if enabled
        if request.enable_validation:
            validation_result = await response_validator.validate_response(
                response=llm_response,
                query=request.query,
                context=request.context,
                retrieved_documents=reranked_results
            )
            
            if not validation_result.is_valid:
                logger.warning(f"Response validation failed: {validation_result.errors}")
        
        # Step 6: Format response
        sources = []
        for doc in reranked_results:
            sources.append(SourceDocument(
                id=doc.get("id"),
                content=doc.get("content", "")[:500] + "..." if len(doc.get("content", "")) > 500 else doc.get("content", ""),
                document_type=doc.get("document_type", "unknown"),
                metadata=doc.get("metadata", {}),
                relevance_score=doc.get("relevance_score", 0.0)
            ))
        
        # Calculate confidence score
        confidence = _calculate_confidence_score(reranked_results, llm_response)
        
        # Extract suggested actions
        suggested_actions = _extract_suggested_actions(llm_response)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Track completion
        research_instrumentation.track_query_latency("query.complete", latency_ms)
        
        # Store query for research analysis
        background_tasks.add_task(
            _store_query_for_research,
            request.query,
            request.context,
            reranked_results,
            llm_response,
            latency_ms,
            trace_id
        )
        
        return QueryResponse(
            answer=llm_response.get("answer", ""),
            sources=sources,
            confidence=confidence,
            suggested_actions=suggested_actions,
            trace_id=trace_id,
            latency_ms=latency_ms,
            retrieval_metadata={
                "retrieval_count": len(retrieval_results),
                "reranked_count": len(reranked_results),
                "avg_relevance_score": sum(doc.get("relevance_score", 0) for doc in reranked_results) / max(len(reranked_results), 1)
            }
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        research_instrumentation.track_query_latency("query.error", int((time.time() - start_time) * 1000))
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/query/batch")
async def batch_query(
    queries: List[QueryRequest],
    db: Session = Depends(get_db)
) -> List[QueryResponse]:
    """
    Process multiple queries in batch for research evaluation.
    """
    results = []
    
    for query_request in queries:
        try:
            result = await query_documents(query_request, None, db)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch query failed for query: {query_request.query}")
            # Add error response
            results.append(QueryResponse(
                answer=f"Error: {str(e)}",
                sources=[],
                confidence=0.0,
                suggested_actions=[],
                trace_id=str(uuid.uuid4()),
                latency_ms=0,
                retrieval_metadata={}
            ))
    
    return results


@router.get("/query/suggestions")
async def get_query_suggestions(
    partial_query: str = "",
    context: Dict[str, Any] = None,
    limit: int = 10
) -> List[str]:
    """
    Get query suggestions based on partial input and context.
    """
    try:
        # This would typically use a suggestion service
        # For now, return some basic suggestions
        suggestions = [
            "How to troubleshoot 503 errors?",
            "What are the steps to restart a service?",
            "How to check service health?",
            "What commands to run for debugging?",
            "How to rollback a deployment?"
        ]
        
        if partial_query:
            # Filter suggestions based on partial query
            suggestions = [s for s in suggestions if partial_query.lower() in s.lower()]
        
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get query suggestions: {e}")
        return []


@router.get("/query/analytics")
async def get_query_analytics(
    time_range: str = "24h",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get query analytics for research monitoring.
    """
    try:
        # Get query statistics from database
        from ...core.database import db_manager
        
        # This would typically query the database for analytics
        # For now, return mock data
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "success_rate": 1.0,
            "top_queries": [],
            "query_types": {},
            "performance_metrics": {
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "avg_retrieval_count": 0,
                "avg_confidence_score": 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get query analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve query analytics")


def _calculate_confidence_score(documents: List[Dict[str, Any]], llm_response: Dict[str, Any]) -> float:
    """Calculate confidence score based on retrieval and generation quality."""
    if not documents:
        return 0.0
    
    # Base confidence from retrieval quality
    avg_relevance = sum(doc.get("relevance_score", 0) for doc in documents) / len(documents)
    
    # Adjust based on number of sources
    source_bonus = min(len(documents) / 5, 0.2)  # Bonus for more sources
    
    # Adjust based on LLM confidence if available
    llm_confidence = llm_response.get("confidence", 0.8)
    
    confidence = (avg_relevance * 0.6 + llm_confidence * 0.4) + source_bonus
    return min(confidence, 1.0)


def _extract_suggested_actions(llm_response: Dict[str, Any]) -> List[str]:
    """Extract suggested actions from LLM response."""
    actions = []
    
    # Look for action patterns in the response
    response_text = llm_response.get("answer", "")
    
    # Simple pattern matching for actions
    import re
    action_patterns = [
        r"1\.\s*([^\\n]+)",
        r"Step \d+:\s*([^\\n]+)",
        r"Action:\s*([^\\n]+)",
        r"Run:\s*([^\\n]+)"
    ]
    
    for pattern in action_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        actions.extend(matches)
    
    return actions[:5]  # Limit to 5 actions


async def _store_query_for_research(
    query: str,
    context: Dict[str, Any],
    retrieved_documents: List[Dict[str, Any]],
    llm_response: Dict[str, Any],
    latency_ms: int,
    trace_id: str
):
    """Store query data for research analysis."""
    try:
        from ...core.database import SessionLocal
        from ...models.database import Query
        
        with SessionLocal() as db:
            query_record = Query(
                id=trace_id,
                query_text=query,
                context=context,
                retrieved_chunks=[doc.get("id") for doc in retrieved_documents],
                llm_response=llm_response.get("answer", ""),
                latency_ms=latency_ms
            )
            db.add(query_record)
            db.commit()
            
    except Exception as e:
        logger.error(f"Failed to store query for research: {e}")
