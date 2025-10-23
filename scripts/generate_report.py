#!/usr/bin/env python3
"""
📊 Generate Mind-Blowing Performance Report
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from pathlib import Path
import sys

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceReporter:
    """📊 Generate comprehensive performance reports."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
    
    async def generate_report(self):
        """📊 Generate the ultimate performance report."""
        print("📊" + "="*80)
        print("📊 GENERATING MIND-BLOWING PERFORMANCE REPORT")
        print("📊" + "="*80)
        print()
        
        # Collect all metrics
        metrics = await self._collect_metrics()
        
        # Generate report
        report = await self._create_report(metrics)
        
        # Save report
        await self._save_report(report)
        
        # Display summary
        await self._display_summary(report)
        
        return report
    
    async def _collect_metrics(self):
        """📈 Collect all system metrics."""
        print("📈 Collecting system metrics...")
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_health": {},
            "performance": {},
            "usage": {},
            "research": {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # System health
                async with session.get(f"{self.api_base_url}/api/v1/health") as response:
                    if response.status == 200:
                        metrics["system_health"] = await response.json()
                
                # System stats
                async with session.get(f"{self.api_base_url}/api/v1/health/stats") as response:
                    if response.status == 200:
                        metrics["usage"] = await response.json()
                
                # Query analytics
                async with session.get(f"{self.api_base_url}/api/v1/query/analytics") as response:
                    if response.status == 200:
                        metrics["performance"] = await response.json()
                
                # Feedback stats
                async with session.get(f"{self.api_base_url}/api/v1/feedback/stats") as response:
                    if response.status == 200:
                        metrics["research"] = await response.json()
                        
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    async def _create_report(self, metrics):
        """📋 Create comprehensive report."""
        print("📋 Creating comprehensive report...")
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "performance_analysis",
                "version": "1.0.0"
            },
            "executive_summary": {
                "system_status": metrics["system_health"].get("status", "unknown"),
                "total_documents": metrics["usage"].get("total_documents", 0),
                "total_queries": metrics["performance"].get("total_queries", 0),
                "success_rate": metrics["performance"].get("success_rate", 0),
                "avg_latency": metrics["performance"].get("avg_latency_ms", 0)
            },
            "performance_metrics": {
                "latency": {
                    "avg_ms": metrics["performance"].get("avg_latency_ms", 0),
                    "p95_ms": metrics["performance"].get("performance_metrics", {}).get("p95_latency_ms", 0),
                    "p99_ms": metrics["performance"].get("performance_metrics", {}).get("p99_latency_ms", 0)
                },
                "accuracy": {
                    "success_rate": metrics["performance"].get("success_rate", 0),
                    "avg_confidence": metrics["performance"].get("performance_metrics", {}).get("avg_confidence_score", 0)
                },
                "throughput": {
                    "queries_per_hour": metrics["performance"].get("total_queries", 0) * 24,  # Rough estimate
                    "avg_retrieval_count": metrics["performance"].get("performance_metrics", {}).get("avg_retrieval_count", 0)
                }
            },
            "system_health": {
                "database_healthy": metrics["system_health"].get("database_healthy", False),
                "redis_healthy": metrics["system_health"].get("redis_healthy", False),
                "vector_extension": metrics["system_health"].get("vector_extension_available", False)
            },
            "usage_statistics": {
                "documents": {
                    "total": metrics["usage"].get("total_documents", 0),
                    "by_type": metrics["usage"].get("documents_by_type", {})
                },
                "chunks": {
                    "total": metrics["usage"].get("total_chunks", 0),
                    "embeddings": metrics["usage"].get("total_embeddings", 0)
                },
                "queries": {
                    "total": metrics["usage"].get("total_queries", 0),
                    "with_feedback": metrics["usage"].get("queries_with_feedback", 0),
                    "avg_feedback": metrics["usage"].get("average_feedback_score", 0)
                }
            },
            "research_insights": {
                "feedback_quality": {
                    "total_feedback": metrics["research"].get("total_feedback", 0),
                    "avg_score": metrics["research"].get("average_score", 0),
                    "quality_rate": metrics["research"].get("quality_rate", 0)
                },
                "user_satisfaction": {
                    "high_quality_responses": metrics["research"].get("high_quality_responses", 0),
                    "low_quality_responses": metrics["research"].get("low_quality_responses", 0)
                }
            },
            "recommendations": [
                "Continue monitoring query latency for optimal performance",
                "Analyze feedback patterns to improve response quality",
                "Consider scaling resources based on usage growth",
                "Implement additional monitoring for production deployment"
            ]
        }
        
        return report
    
    async def _save_report(self, report):
        """💾 Save report to file."""
        print("💾 Saving report to file...")
        
        report_path = Path("performance_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {report_path}")
    
    async def _display_summary(self, report):
        """📊 Display report summary."""
        print("📊" + "="*80)
        print("📊 PERFORMANCE REPORT SUMMARY")
        print("📊" + "="*80)
        print()
        
        # Executive Summary
        exec_summary = report["executive_summary"]
        print("🎯 EXECUTIVE SUMMARY:")
        print(f"   🚀 System Status: {exec_summary['system_status'].upper()}")
        print(f"   📚 Total Documents: {exec_summary['total_documents']:,}")
        print(f"   💬 Total Queries: {exec_summary['total_queries']:,}")
        print(f"   ✅ Success Rate: {exec_summary['success_rate']:.1%}")
        print(f"   ⚡ Avg Latency: {exec_summary['avg_latency']:.0f}ms")
        print()
        
        # Performance Metrics
        perf = report["performance_metrics"]
        print("⚡ PERFORMANCE METRICS:")
        print(f"   🎯 Average Latency: {perf['latency']['avg_ms']:.0f}ms")
        print(f"   🎯 P95 Latency: {perf['latency']['p95_ms']:.0f}ms")
        print(f"   🎯 P99 Latency: {perf['latency']['p99_ms']:.0f}ms")
        print(f"   ✅ Success Rate: {perf['accuracy']['success_rate']:.1%}")
        print(f"   🎯 Avg Confidence: {perf['accuracy']['avg_confidence']:.1%}")
        print()
        
        # Usage Statistics
        usage = report["usage_statistics"]
        print("📊 USAGE STATISTICS:")
        print(f"   📚 Documents: {usage['documents']['total']:,}")
        print(f"   🔍 Chunks: {usage['chunks']['total']:,}")
        print(f"   🧠 Embeddings: {usage['chunks']['embeddings']:,}")
        print(f"   💬 Queries: {usage['queries']['total']:,}")
        print()
        
        # Research Insights
        research = report["research_insights"]
        print("🔬 RESEARCH INSIGHTS:")
        print(f"   ⭐ Total Feedback: {research['feedback_quality']['total_feedback']:,}")
        print(f"   📊 Avg Score: {research['feedback_quality']['avg_score']:.1f}/5.0")
        print(f"   🎯 Quality Rate: {research['feedback_quality']['quality_rate']:.1%}")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
        print()
        
        print("🎉" + "="*80)
        print("🎉 REPORT GENERATION COMPLETE!")
        print("🎉" + "="*80)


async def main():
    """📊 Main report generation function."""
    reporter = PerformanceReporter()
    await reporter.generate_report()


if __name__ == "__main__":
    asyncio.run(main())
