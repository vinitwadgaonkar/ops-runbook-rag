"""
Feedback API endpoint for user ratings and research data collection.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import uuid

from ...core.database import get_db
from ...models.schemas import FeedbackRequest, FeedbackResponse
from ...models.database import Query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
) -> FeedbackResponse:
    """
    Submit user feedback for a query response.
    
    This endpoint collects:
    - User ratings (1-5 scale)
    - Feedback text
    - Suggested improvements
    - Query context for research analysis
    """
    try:
        # Validate query exists
        query_record = db.query(Query).filter(Query.id == feedback.query_id).first()
        if not query_record:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Update query with feedback
        query_record.feedback_score = feedback.feedback_score
        db.commit()
        
        # Create feedback response
        feedback_response = FeedbackResponse(
            id=str(uuid.uuid4()),
            query_id=feedback.query_id,
            feedback_score=feedback.feedback_score,
            feedback_text=feedback.feedback_text,
            created_at=datetime.utcnow()
        )
        
        # Store additional feedback data in metadata
        if feedback.suggested_improvements:
            # This would typically be stored in a separate feedback table
            # For now, we'll log it for research purposes
            logger.info(f"Feedback improvements for query {feedback.query_id}: {feedback.suggested_improvements}")
        
        logger.info(f"Feedback submitted for query {feedback.query_id}: score={feedback.feedback_score}")
        
        return feedback_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.get("/feedback/stats")
async def get_feedback_stats(
    time_range: str = "7d",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get feedback statistics for research analysis.
    """
    try:
        # Get feedback statistics from database
        from sqlalchemy import func, desc
        
        # Total feedback count
        total_feedback = db.query(Query).filter(Query.feedback_score.isnot(None)).count()
        
        # Average feedback score
        avg_score = db.query(func.avg(Query.feedback_score)).filter(
            Query.feedback_score.isnot(None)
        ).scalar() or 0.0
        
        # Feedback distribution
        score_distribution = {}
        for score in range(1, 6):
            count = db.query(Query).filter(Query.feedback_score == score).count()
            score_distribution[f"score_{score}"] = count
        
        # Recent feedback (last 7 days)
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        recent_feedback = db.query(Query).filter(
            Query.feedback_score.isnot(None),
            Query.created_at >= cutoff_date
        ).count()
        
        # High-quality responses (score >= 4)
        high_quality_count = db.query(Query).filter(
            Query.feedback_score >= 4
        ).count()
        
        # Low-quality responses (score <= 2)
        low_quality_count = db.query(Query).filter(
            Query.feedback_score <= 2
        ).count()
        
        return {
            "total_feedback": total_feedback,
            "average_score": round(avg_score, 2),
            "score_distribution": score_distribution,
            "recent_feedback_7d": recent_feedback,
            "high_quality_responses": high_quality_count,
            "low_quality_responses": low_quality_count,
            "quality_rate": high_quality_count / max(total_feedback, 1),
            "time_range": time_range
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback statistics")


@router.get("/feedback/analytics")
async def get_feedback_analytics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed feedback analytics for research insights.
    """
    try:
        from sqlalchemy import func, desc
        from datetime import timedelta
        
        # Get feedback trends over time
        feedback_trends = []
        for days_back in range(7, 0, -1):
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            next_day = cutoff_date + timedelta(days=1)
            
            day_feedback = db.query(Query).filter(
                Query.feedback_score.isnot(None),
                Query.created_at >= cutoff_date,
                Query.created_at < next_day
            ).all()
            
            if day_feedback:
                avg_score = sum(q.feedback_score for q in day_feedback) / len(day_feedback)
                feedback_trends.append({
                    "date": cutoff_date.strftime("%Y-%m-%d"),
                    "count": len(day_feedback),
                    "avg_score": round(avg_score, 2)
                })
        
        # Get correlation between latency and feedback score
        latency_feedback = db.query(
            Query.latency_ms, Query.feedback_score
        ).filter(
            Query.feedback_score.isnot(None),
            Query.latency_ms.isnot(None)
        ).all()
        
        if latency_feedback:
            # Calculate correlation (simplified)
            latencies = [item[0] for item in latency_feedback]
            scores = [item[1] for item in latency_feedback]
            
            avg_latency = sum(latencies) / len(latencies)
            avg_score = sum(scores) / len(scores)
            
            # Simple correlation calculation
            correlation = 0.0  # Would need proper correlation calculation
        else:
            avg_latency = 0
            avg_score = 0
            correlation = 0.0
        
        # Get feedback by query characteristics
        feedback_by_type = {}
        # This would analyze feedback by query type, context, etc.
        
        return {
            "feedback_trends": feedback_trends,
            "latency_correlation": {
                "avg_latency_ms": avg_latency,
                "avg_score": avg_score,
                "correlation": correlation
            },
            "feedback_by_type": feedback_by_type,
            "insights": {
                "total_analyses": len(latency_feedback),
                "trend_period": "7 days",
                "data_quality": "good" if len(latency_feedback) > 10 else "limited"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get feedback analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback analytics")


@router.get("/feedback/queries/{query_id}")
async def get_query_feedback(
    query_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get feedback for a specific query.
    """
    try:
        query_record = db.query(Query).filter(Query.id == query_id).first()
        
        if not query_record:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "query_id": query_id,
            "query_text": query_record.query_text,
            "feedback_score": query_record.feedback_score,
            "created_at": query_record.created_at,
            "latency_ms": query_record.latency_ms,
            "has_feedback": query_record.feedback_score is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get query feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve query feedback")


@router.get("/feedback/export")
async def export_feedback_data(
    format: str = "json",
    time_range: str = "30d",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Export feedback data for research analysis.
    """
    try:
        from datetime import timedelta
        
        # Calculate date range
        if time_range.endswith('d'):
            days = int(time_range[:-1])
        else:
            days = 30
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get feedback data
        feedback_data = db.query(Query).filter(
            Query.feedback_score.isnot(None),
            Query.created_at >= cutoff_date
        ).all()
        
        # Format data for export
        export_data = []
        for query in feedback_data:
            export_data.append({
                "query_id": str(query.id),
                "query_text": query.query_text,
                "feedback_score": query.feedback_score,
                "latency_ms": query.latency_ms,
                "created_at": query.created_at.isoformat(),
                "context": query.context
            })
        
        return {
            "format": format,
            "time_range": time_range,
            "record_count": len(export_data),
            "data": export_data,
            "exported_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to export feedback data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export feedback data")


@router.post("/feedback/bulk")
async def bulk_feedback_import(
    feedback_data: List[Dict[str, Any]],
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Import bulk feedback data for research purposes.
    """
    try:
        imported_count = 0
        errors = []
        
        for feedback_item in feedback_data:
            try:
                query_id = feedback_item.get("query_id")
                feedback_score = feedback_item.get("feedback_score")
                
                if not query_id or not feedback_score:
                    errors.append(f"Missing required fields: {feedback_item}")
                    continue
                
                # Update query with feedback
                query_record = db.query(Query).filter(Query.id == query_id).first()
                if query_record:
                    query_record.feedback_score = feedback_score
                    imported_count += 1
                else:
                    errors.append(f"Query not found: {query_id}")
                    
            except Exception as e:
                errors.append(f"Error processing {feedback_item}: {str(e)}")
        
        db.commit()
        
        return {
            "imported_count": imported_count,
            "error_count": len(errors),
            "errors": errors[:10],  # Limit error details
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Bulk feedback import failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to import bulk feedback data")
