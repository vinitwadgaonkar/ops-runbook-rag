#!/usr/bin/env python3
"""
Evaluation script for the Ops Runbook RAG system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import aiohttp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for RAG system performance."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def evaluate_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query against the RAG system.
        
        Args:
            query_data: Query data from evaluation dataset
            
        Returns:
            Evaluation results
        """
        try:
            # Make query to RAG system
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query_data["query"],
                    "context": query_data.get("context", {}),
                    "max_results": 5,
                    "include_sources": True
                }
                
                async with session.post(
                    f"{self.api_base_url}/api/v1/query",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Query failed with status {response.status}")
                        return {"error": f"HTTP {response.status}"}
                    
                    result = await response.json()
            
            # Calculate evaluation metrics
            metrics = await self._calculate_metrics(
                query_data["expected_answer"],
                result["answer"],
                result.get("sources", [])
            )
            
            return {
                "query_id": query_data["id"],
                "query": query_data["query"],
                "expected_answer": query_data["expected_answer"],
                "actual_answer": result["answer"],
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "latency_ms": result.get("latency_ms", 0),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed for query {query_data['id']}: {e}")
            return {
                "query_id": query_data["id"],
                "error": str(e),
                "metrics": {
                    "bleu_score": 0.0,
                    "rouge_l": 0.0,
                    "semantic_similarity": 0.0,
                    "actionability_score": 0
                }
            }
    
    async def _calculate_metrics(self, expected: str, actual: str, sources: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # BLEU Score (simplified)
        metrics["bleu_score"] = self._calculate_bleu(expected, actual)
        
        # ROUGE-L Score (simplified)
        metrics["rouge_l"] = self._calculate_rouge_l(expected, actual)
        
        # Semantic Similarity
        metrics["semantic_similarity"] = self._calculate_semantic_similarity(expected, actual)
        
        # Actionability Score (simplified heuristic)
        metrics["actionability_score"] = self._calculate_actionability_score(actual)
        
        # Source Relevance
        metrics["source_relevance"] = self._calculate_source_relevance(sources)
        
        return metrics
    
    def _calculate_bleu(self, expected: str, actual: str) -> float:
        """Calculate BLEU score (simplified implementation)."""
        # This is a simplified BLEU calculation
        # In practice, you'd use a proper BLEU implementation
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words.intersection(actual_words))
        precision = overlap / len(actual_words) if actual_words else 0.0
        
        return min(precision, 1.0)
    
    def _calculate_rouge_l(self, expected: str, actual: str) -> float:
        """Calculate ROUGE-L score (simplified implementation)."""
        # This is a simplified ROUGE-L calculation
        # In practice, you'd use a proper ROUGE implementation
        expected_words = expected.lower().split()
        actual_words = actual.lower().split()
        
        if not expected_words:
            return 0.0
        
        # Simple word overlap
        overlap = 0
        for word in expected_words:
            if word in actual_words:
                overlap += 1
        
        recall = overlap / len(expected_words)
        return recall
    
    def _calculate_semantic_similarity(self, expected: str, actual: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        try:
            embeddings = self.similarity_model.encode([expected, actual])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_actionability_score(self, answer: str) -> float:
        """Calculate actionability score based on answer content."""
        answer_lower = answer.lower()
        
        # Look for actionable indicators
        action_indicators = [
            "check", "verify", "run", "execute", "restart", "scale",
            "kubectl", "docker", "curl", "command", "step", "procedure"
        ]
        
        score = 0.0
        for indicator in action_indicators:
            if indicator in answer_lower:
                score += 0.1
        
        # Look for command patterns
        if "```" in answer or "$" in answer:
            score += 0.3
        
        # Look for step-by-step structure
        if any(word in answer_lower for word in ["1.", "2.", "step", "first", "then", "next"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_source_relevance(self, sources: List[Dict]) -> float:
        """Calculate source relevance score."""
        if not sources:
            return 0.0
        
        # Simple relevance based on number of sources and their scores
        total_relevance = sum(source.get("relevance_score", 0) for source in sources)
        avg_relevance = total_relevance / len(sources)
        
        # Bonus for having multiple sources
        source_bonus = min(len(sources) / 5, 0.2)
        
        return min(avg_relevance + source_bonus, 1.0)
    
    async def evaluate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate the entire dataset.
        
        Args:
            dataset_path: Path to evaluation dataset JSON file
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Loading evaluation dataset from {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        incidents = dataset.get("incidents", [])
        logger.info(f"Evaluating {len(incidents)} queries...")
        
        results = []
        for incident in incidents:
            result = await self.evaluate_query(incident)
            results.append(result)
            logger.info(f"Evaluated query {incident['id']}")
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(results)
        
        return {
            "dataset_info": {
                "name": dataset.get("dataset_name", "Unknown"),
                "version": dataset.get("version", "1.0.0"),
                "total_queries": len(incidents)
            },
            "results": results,
            "metrics": metrics,
            "comparison": self._compare_with_baseline(metrics, dataset.get("baseline_metrics", {}))
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results."""
        valid_results = [r for r in results if "metrics" in r and "error" not in r]
        
        if not valid_results:
            return {
                "avg_bleu_score": 0.0,
                "avg_rouge_l": 0.0,
                "avg_semantic_similarity": 0.0,
                "avg_actionability_score": 0.0,
                "avg_source_relevance": 0.0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "success_rate": 0.0
            }
        
        metrics = {}
        for metric in ["bleu_score", "rouge_l", "semantic_similarity", "actionability_score", "source_relevance"]:
            values = [r["metrics"][metric] for r in valid_results if metric in r["metrics"]]
            metrics[f"avg_{metric}"] = np.mean(values) if values else 0.0
        
        # Additional metrics
        metrics["avg_confidence"] = np.mean([r.get("confidence", 0) for r in valid_results])
        metrics["avg_latency_ms"] = np.mean([r.get("latency_ms", 0) for r in valid_results])
        metrics["success_rate"] = len(valid_results) / len(results)
        
        return metrics
    
    def _compare_with_baseline(self, metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {}
        
        for metric, current_value in metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                improvement = current_value - baseline_value
                improvement_pct = (improvement / baseline_value * 100) if baseline_value > 0 else 0
                
                comparison[metric] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct
                }
        
        return comparison


async def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset JSON")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(args.api_url)
    
    try:
        # Run evaluation
        results = await evaluator.evaluate_dataset(args.dataset)
        
        # Print results
        logger.info("📊 Evaluation Results:")
        logger.info(f"  Dataset: {results['dataset_info']['name']} v{results['dataset_info']['version']}")
        logger.info(f"  Total queries: {results['dataset_info']['total_queries']}")
        logger.info(f"  Success rate: {results['metrics']['success_rate']:.2%}")
        
        logger.info("\n📈 Performance Metrics:")
        for metric, value in results['metrics'].items():
            if metric.startswith('avg_'):
                logger.info(f"  {metric}: {value:.3f}")
        
        # Show improvements over baseline
        if results['comparison']:
            logger.info("\n🚀 Improvements over Baseline:")
            for metric, comparison in results['comparison'].items():
                if comparison['improvement_pct'] > 0:
                    logger.info(f"  {metric}: +{comparison['improvement_pct']:.1f}%")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📁 Results saved to {args.output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
