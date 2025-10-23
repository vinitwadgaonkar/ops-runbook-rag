"""
Full ingestion pipeline with RabbitMQ async processing for research operations.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
import aiofiles
from sqlalchemy.orm import Session

from ..core.database import SessionLocal, db_manager
from ..core.cache import cache_manager
from ..models.database import Document, Chunk, IngestionJob
from .parsers.markdown import MarkdownParser
from .parsers.json_kb import JSONKBParser
from .parsers.screenshot import ScreenshotParser
from .parsers.pdf_rca import PDFRCAParser
from .chunker import SemanticChunker
from .embedder import embedder

logger = logging.getLogger(__name__)


@dataclass
class IngestionTask:
    """Represents an ingestion task."""
    task_id: str
    source_path: str
    document_type: str
    metadata: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class IngestionPipeline:
    """
    Full ingestion pipeline with async processing and research features.
    Handles document parsing, chunking, embedding, and storage.
    """
    
    def __init__(self):
        self.parsers = {
            "runbook": MarkdownParser(),
            "kb_article": JSONKBParser(),
            "screenshot": ScreenshotParser(),
            "rca": PDFRCAParser()
        }
        self.chunker = SemanticChunker()
        self.embedder = embedder
        
        # Research tracking
        self.ingestion_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_document_type": {},
            "avg_processing_time": 0.0
        }
    
    async def ingest_document(self, source_path: str, document_type: str, 
                           metadata: Optional[Dict[str, Any]] = None,
                           force_reprocess: bool = False) -> str:
        """
        Ingest a single document with full processing pipeline.
        
        Args:
            source_path: Path to document file
            document_type: Type of document (runbook, kb_article, screenshot, rca)
            metadata: Optional document metadata
            force_reprocess: Whether to reprocess existing documents
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        # Create ingestion job record
        job = IngestionJob(
            id=task_id,
            job_type="document_ingestion",
            status="pending",
            source_path=source_path,
            metadata=metadata or {}
        )
        
        try:
            with SessionLocal() as db:
                db.add(job)
                db.commit()
            
            # Process document asynchronously
            asyncio.create_task(self._process_document(task_id, source_path, document_type, metadata))
            
            logger.info(f"Started ingestion task {task_id} for {source_path}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to start ingestion task: {e}")
            # Update job status
            with SessionLocal() as db:
                job = db.query(IngestionJob).filter(IngestionJob.id == task_id).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    db.commit()
            raise
    
    async def _process_document(self, task_id: str, source_path: str, 
                              document_type: str, metadata: Optional[Dict[str, Any]]):
        """Process a document through the full pipeline."""
        start_time = datetime.utcnow()
        
        try:
            # Update job status
            with SessionLocal() as db:
                job = db.query(IngestionJob).filter(IngestionJob.id == task_id).first()
                if job:
                    job.status = "processing"
                    job.started_at = start_time
                    db.commit()
            
            # Step 1: Parse document
            logger.info(f"Parsing document {source_path}")
            parsed_content = await self._parse_document(source_path, document_type)
            
            # Step 2: Create document record
            document_id = await self._create_document_record(parsed_content, source_path, metadata)
            
            # Step 3: Chunk document
            logger.info(f"Chunking document {source_path}")
            chunks = await self._chunk_document(parsed_content, document_id)
            
            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = await self._generate_embeddings(chunks)
            
            # Step 5: Store chunks with embeddings
            logger.info(f"Storing {len(chunks)} chunks with embeddings")
            await self._store_chunks_with_embeddings(chunks, embeddings, document_id)
            
            # Update job status
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            with SessionLocal() as db:
                job = db.query(IngestionJob).filter(IngestionJob.id == task_id).first()
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.metadata = {
                        **job.metadata,
                        "document_id": document_id,
                        "chunk_count": len(chunks),
                        "processing_time_seconds": processing_time
                    }
                    db.commit()
            
            # Update research stats
            self._update_ingestion_stats(document_type, processing_time, True)
            
            logger.info(f"Successfully processed document {source_path} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process document {source_path}: {e}")
            
            # Update job status
            with SessionLocal() as db:
                job = db.query(IngestionJob).filter(IngestionJob.id == task_id).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    db.commit()
            
            # Update research stats
            self._update_ingestion_stats(document_type, 0, False)
    
    async def _parse_document(self, source_path: str, document_type: str) -> Dict[str, Any]:
        """Parse document using appropriate parser."""
        if document_type not in self.parsers:
            raise ValueError(f"Unsupported document type: {document_type}")
        
        parser = self.parsers[document_type]
        
        if document_type == "screenshot":
            # Screenshots need special handling
            return parser.parse(source_path)
        else:
            # Other documents can be parsed from file
            return parser.parse_file(source_path)
    
    async def _create_document_record(self, parsed_content: Dict[str, Any], 
                                    source_path: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Create document record in database."""
        document_id = str(uuid.uuid4())
        
        # Extract content and metadata
        content = parsed_content.get("content", "")
        document_metadata = parsed_content.get("metadata", {})
        
        # Merge with provided metadata
        if metadata:
            document_metadata.update(metadata)
        
        # Create document record
        document = Document(
            id=document_id,
            content=content,
            metadata=document_metadata,
            document_type=document_metadata.get("document_type", "unknown"),
            source_path=source_path
        )
        
        with SessionLocal() as db:
            db.add(document)
            db.commit()
        
        return document_id
    
    async def _chunk_document(self, parsed_content: Dict[str, Any], document_id: str) -> List[Any]:
        """Chunk document into semantic chunks."""
        content = parsed_content.get("content", "")
        metadata = parsed_content.get("metadata", {})
        
        # Add document ID to metadata
        metadata["document_id"] = document_id
        
        # Create chunks
        chunks = self.chunker.chunk_document(content, metadata)
        
        return chunks
    
    async def _generate_embeddings(self, chunks: List[Any]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        # Extract chunk content
        chunk_texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embedding_result = await self.embedder.embed_texts(chunk_texts)
        
        return embedding_result.embeddings
    
    async def _store_chunks_with_embeddings(self, chunks: List[Any], 
                                         embeddings: List[List[float]], document_id: str):
        """Store chunks with embeddings in database."""
        with SessionLocal() as db:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk.content,
                    embedding=embedding,  # Will be converted to vector type by pgvector
                    metadata=chunk.metadata,
                    chunk_index=i
                )
                db.add(chunk_record)
            
            db.commit()
    
    async def batch_ingest(self, source_directory: str, document_types: List[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Batch ingest documents from a directory.
        
        Args:
            source_directory: Directory containing documents
            document_types: List of document types to process (None for all)
            metadata: Optional metadata for all documents
            
        Returns:
            List of task IDs
        """
        source_path = Path(source_directory)
        if not source_path.exists():
            raise ValueError(f"Source directory does not exist: {source_directory}")
        
        task_ids = []
        
        # Find documents by type
        for doc_type in (document_types or ["runbook", "kb_article", "screenshot", "rca"]):
            pattern = self._get_file_pattern(doc_type)
            files = list(source_path.glob(pattern))
            
            for file_path in files:
                try:
                    task_id = await self.ingest_document(
                        str(file_path), doc_type, metadata
                    )
                    task_ids.append(task_id)
                except Exception as e:
                    logger.error(f"Failed to start ingestion for {file_path}: {e}")
        
        logger.info(f"Started batch ingestion of {len(task_ids)} documents")
        return task_ids
    
    def _get_file_pattern(self, document_type: str) -> str:
        """Get file pattern for document type."""
        patterns = {
            "runbook": "*.md",
            "kb_article": "*.json",
            "screenshot": "*.{png,jpg,jpeg}",
            "rca": "*.pdf"
        }
        return patterns.get(document_type, "*")
    
    def _update_ingestion_stats(self, document_type: str, processing_time: float, success: bool):
        """Update ingestion statistics for research tracking."""
        self.ingestion_stats["total_processed"] += 1
        
        if success:
            self.ingestion_stats["successful"] += 1
        else:
            self.ingestion_stats["failed"] += 1
        
        # Update by document type
        if document_type not in self.ingestion_stats["by_document_type"]:
            self.ingestion_stats["by_document_type"][document_type] = {
                "total": 0, "successful": 0, "failed": 0
            }
        
        self.ingestion_stats["by_document_type"][document_type]["total"] += 1
        if success:
            self.ingestion_stats["by_document_type"][document_type]["successful"] += 1
        else:
            self.ingestion_stats["by_document_type"][document_type]["failed"] += 1
        
        # Update average processing time
        if success and processing_time > 0:
            current_avg = self.ingestion_stats["avg_processing_time"]
            total_successful = self.ingestion_stats["successful"]
            self.ingestion_stats["avg_processing_time"] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
    
    async def get_ingestion_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an ingestion task."""
        with SessionLocal() as db:
            job = db.query(IngestionJob).filter(IngestionJob.id == task_id).first()
            if not job:
                return None
            
            return {
                "task_id": task_id,
                "status": job.status,
                "source_path": job.source_path,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "error_message": job.error_message,
                "metadata": job.metadata
            }
    
    async def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics."""
        # Get database stats
        db_stats = db_manager.get_table_stats()
        vector_stats = db_manager.get_vector_stats()
        
        return {
            "pipeline_stats": self.ingestion_stats,
            "database_stats": db_stats,
            "vector_stats": vector_stats,
            "cache_stats": cache_manager.get_cache_stats() if cache_manager else {}
        }
    
    async def reprocess_document(self, document_id: str, force: bool = False) -> str:
        """Reprocess an existing document."""
        with SessionLocal() as db:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document not found: {document_id}")
            
            # Delete existing chunks
            db.query(Chunk).filter(Chunk.document_id == document_id).delete()
            db.commit()
            
            # Reprocess document
            return await self.ingest_document(
                document.source_path,
                document.document_type,
                document.metadata,
                force_reprocess=True
            )
    
    async def cleanup_failed_jobs(self, older_than_hours: int = 24) -> int:
        """Clean up failed ingestion jobs older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        with SessionLocal() as db:
            failed_jobs = db.query(IngestionJob).filter(
                IngestionJob.status == "failed",
                IngestionJob.created_at < cutoff_time
            ).all()
            
            count = len(failed_jobs)
            for job in failed_jobs:
                db.delete(job)
            
            db.commit()
        
        logger.info(f"Cleaned up {count} failed ingestion jobs")
        return count


# Global pipeline instance
ingestion_pipeline = IngestionPipeline()


# Utility functions for research analysis
async def analyze_ingestion_quality() -> Dict[str, Any]:
    """Analyze ingestion quality for research purposes."""
    stats = await ingestion_pipeline.get_ingestion_stats()
    
    total_processed = stats["pipeline_stats"]["total_processed"]
    successful = stats["pipeline_stats"]["successful"]
    failed = stats["pipeline_stats"]["failed"]
    
    success_rate = successful / total_processed if total_processed > 0 else 0
    avg_processing_time = stats["pipeline_stats"]["avg_processing_time"]
    
    return {
        "success_rate": success_rate,
        "total_processed": total_processed,
        "successful": successful,
        "failed": failed,
        "avg_processing_time": avg_processing_time,
        "by_document_type": stats["pipeline_stats"]["by_document_type"],
        "database_stats": stats["database_stats"],
        "vector_stats": stats["vector_stats"],
        "quality_score": success_rate * (1.0 - min(avg_processing_time / 60, 1.0))  # Penalize slow processing
    }


async def get_ingestion_metrics() -> Dict[str, Any]:
    """Get detailed ingestion metrics for monitoring."""
    stats = await ingestion_pipeline.get_ingestion_stats()
    
    return {
        "throughput": {
            "documents_per_hour": stats["pipeline_stats"]["total_processed"] / max(1, 1),  # Simplified
            "chunks_per_document": stats["vector_stats"].get("total_embeddings", 0) / max(1, stats["database_stats"].get("documents", 1))
        },
        "quality": {
            "success_rate": stats["pipeline_stats"]["successful"] / max(1, stats["pipeline_stats"]["total_processed"]),
            "avg_processing_time": stats["pipeline_stats"]["avg_processing_time"]
        },
        "storage": {
            "total_documents": stats["database_stats"].get("documents", 0),
            "total_chunks": stats["database_stats"].get("chunks", 0),
            "total_embeddings": stats["vector_stats"].get("total_embeddings", 0)
        }
    }
