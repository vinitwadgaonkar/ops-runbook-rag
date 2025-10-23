#!/usr/bin/env python3
"""
Batch ingestion script for the Ops Runbook RAG system.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.ingestion.pipeline import ingestion_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_directory(
    data_dir: str,
    document_types: List[str] = None,
    metadata: Dict[str, Any] = None
) -> List[str]:
    """
    Ingest all documents from a directory.
    
    Args:
        data_dir: Directory containing documents
        document_types: List of document types to process
        metadata: Optional metadata for all documents
        
    Returns:
        List of task IDs
    """
    logger.info(f"Starting batch ingestion from {data_dir}")
    
    try:
        task_ids = await ingestion_pipeline.batch_ingest(
            source_directory=data_dir,
            document_types=document_types,
            metadata=metadata
        )
        
        logger.info(f"Started ingestion of {len(task_ids)} documents")
        return task_ids
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise


async def monitor_ingestion_tasks(task_ids: List[str], check_interval: int = 30):
    """
    Monitor ingestion tasks until completion.
    
    Args:
        task_ids: List of task IDs to monitor
        check_interval: Seconds between status checks
    """
    logger.info(f"Monitoring {len(task_ids)} ingestion tasks...")
    
    completed_tasks = set()
    failed_tasks = set()
    
    while len(completed_tasks) + len(failed_tasks) < len(task_ids):
        for task_id in task_ids:
            if task_id in completed_tasks or task_id in failed_tasks:
                continue
                
            try:
                status = await ingestion_pipeline.get_ingestion_status(task_id)
                
                if status:
                    if status["status"] == "completed":
                        completed_tasks.add(task_id)
                        logger.info(f"✅ Task {task_id} completed")
                    elif status["status"] == "failed":
                        failed_tasks.add(task_id)
                        logger.error(f"❌ Task {task_id} failed: {status.get('error_message', 'Unknown error')}")
                    else:
                        logger.info(f"⏳ Task {task_id} status: {status['status']}")
                        
            except Exception as e:
                logger.error(f"Failed to check status for task {task_id}: {e}")
        
        if len(completed_tasks) + len(failed_tasks) < len(task_ids):
            logger.info(f"Waiting {check_interval}s before next check...")
            await asyncio.sleep(check_interval)
    
    logger.info(f"Ingestion completed: {len(completed_tasks)} successful, {len(failed_tasks)} failed")
    return completed_tasks, failed_tasks


async def get_ingestion_stats():
    """Get comprehensive ingestion statistics."""
    try:
        stats = await ingestion_pipeline.get_ingestion_stats()
        
        logger.info("📊 Ingestion Statistics:")
        logger.info(f"  Total processed: {stats['pipeline_stats']['total_processed']}")
        logger.info(f"  Successful: {stats['pipeline_stats']['successful']}")
        logger.info(f"  Failed: {stats['pipeline_stats']['failed']}")
        logger.info(f"  Average processing time: {stats['pipeline_stats']['avg_processing_time']:.2f}s")
        
        logger.info("📈 Database Statistics:")
        for table, count in stats['database_stats'].items():
            logger.info(f"  {table}: {count}")
        
        logger.info("🔢 Vector Statistics:")
        vector_stats = stats['vector_stats']
        logger.info(f"  Total embeddings: {vector_stats.get('total_embeddings', 0)}")
        logger.info(f"  By document type: {vector_stats.get('by_document_type', {})}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get ingestion stats: {e}")
        return None


async def main():
    """Main function for batch ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch ingest documents for RAG system")
    parser.add_argument("--data-dir", required=True, help="Directory containing documents")
    parser.add_argument("--types", nargs="+", help="Document types to process", 
                       choices=["runbook", "kb_article", "screenshot", "rca"])
    parser.add_argument("--metadata", help="JSON metadata for all documents")
    parser.add_argument("--monitor", action="store_true", help="Monitor tasks until completion")
    parser.add_argument("--stats", action="store_true", help="Show ingestion statistics")
    
    args = parser.parse_args()
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        import json
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid metadata JSON: {e}")
            return False
    
    try:
        # Start batch ingestion
        task_ids = await ingest_directory(
            data_dir=args.data_dir,
            document_types=args.types,
            metadata=metadata
        )
        
        if args.monitor:
            # Monitor tasks until completion
            completed, failed = await monitor_ingestion_tasks(task_ids)
            
            if failed:
                logger.warning(f"⚠️  {len(failed)} tasks failed")
                return False
            else:
                logger.info("🎉 All tasks completed successfully!")
        
        if args.stats:
            # Show statistics
            await get_ingestion_stats()
        
        return True
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
