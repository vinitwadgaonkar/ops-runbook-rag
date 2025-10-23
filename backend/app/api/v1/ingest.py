"""
Ingestion API endpoint for document processing.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
import asyncio

from ...core.database import get_db
from ...models.schemas import IngestionRequest, IngestionResponse
from ...ingestion.pipeline import ingestion_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    metadata: Optional[str] = Form(None),
    force_reprocess: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
) -> IngestionResponse:
    """
    Ingest a single document for processing.
    
    Supports:
    - Markdown runbooks (.md)
    - JSON KB articles (.json)
    - Screenshot images (.png, .jpg, .jpeg)
    - PDF RCAs (.pdf)
    """
    try:
        # Validate document type
        valid_types = ["runbook", "kb_article", "screenshot", "rca"]
        if document_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid document type. Must be one of: {valid_types}"
            )
        
        # Parse metadata
        parsed_metadata = {}
        if metadata:
            import json
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Start ingestion task
        task_id = await ingestion_pipeline.ingest_document(
            source_path=temp_path,
            document_type=document_type,
            metadata=parsed_metadata,
            force_reprocess=force_reprocess
        )
        
        # Clean up temp file in background
        if background_tasks:
            background_tasks.add_task(_cleanup_temp_file, temp_path)
        
        return IngestionResponse(
            job_id=task_id,
            status="started",
            message=f"Document ingestion started. Task ID: {task_id}"
        )
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/batch")
async def batch_ingest(
    source_directory: str,
    document_types: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Batch ingest documents from a directory.
    """
    try:
        task_ids = await ingestion_pipeline.batch_ingest(
            source_directory=source_directory,
            document_types=document_types,
            metadata=metadata
        )
        
        return {
            "message": f"Batch ingestion started for {len(task_ids)} documents",
            "task_ids": task_ids,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")


@router.get("/ingest/status/{task_id}")
async def get_ingestion_status(
    task_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get status of an ingestion task.
    """
    try:
        status = await ingestion_pipeline.get_ingestion_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ingestion status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")


@router.get("/ingest/stats")
async def get_ingestion_stats(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive ingestion statistics.
    """
    try:
        stats = await ingestion_pipeline.get_ingestion_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get ingestion stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ingestion statistics")


@router.post("/ingest/reprocess/{document_id}")
async def reprocess_document(
    document_id: str,
    force: bool = False,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Reprocess an existing document.
    """
    try:
        task_id = await ingestion_pipeline.reprocess_document(document_id, force)
        
        return {
            "message": f"Document reprocessing started",
            "task_id": task_id,
            "document_id": document_id,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(f"Document reprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")


@router.delete("/ingest/cleanup")
async def cleanup_failed_jobs(
    older_than_hours: int = 24,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Clean up failed ingestion jobs.
    """
    try:
        cleaned_count = await ingestion_pipeline.cleanup_failed_jobs(older_than_hours)
        
        return {
            "message": f"Cleaned up {cleaned_count} failed jobs",
            "cleaned_count": cleaned_count,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/ingest/supported-types")
async def get_supported_types() -> Dict[str, Any]:
    """
    Get supported document types and their file extensions.
    """
    return {
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


async def _cleanup_temp_file(file_path: str):
    """Clean up temporary file."""
    try:
        import os
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")
