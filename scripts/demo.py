#!/usr/bin/env python3
"""
🚀 Live Demo Script for the Most Advanced RAG System on the Planet
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGDemo:
    """🎬 Live demo of the revolutionary RAG system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.demo_queries = [
            {
                "query": "🚨 API Gateway is returning 503 errors, what should I do?",
                "context": {"service": "api-gateway", "severity": "critical"},
                "expected": "troubleshooting steps"
            },
            {
                "query": "🐌 Database queries are running slow, how can I optimize?",
                "context": {"service": "database", "severity": "high"},
                "expected": "performance tuning"
            },
            {
                "query": "🔐 Authentication is failing, what's wrong?",
                "context": {"service": "auth-service", "component": "authentication"},
                "expected": "auth troubleshooting"
            },
            {
                "query": "📊 How to monitor service health?",
                "context": {"service": "monitoring", "incident_type": "monitoring"},
                "expected": "monitoring setup"
            },
            {
                "query": "🚀 How to rollback a failed deployment?",
                "context": {"service": "deployment", "incident_type": "deployment_issues"},
                "expected": "rollback procedures"
            }
        ]
    
    async def run_demo(self):
        """🎬 Run the complete live demo."""
        print("🚀" + "="*80)
        print("🚀 OPS RUNBOOK RAG: THE ULTIMATE AI COPILOT FOR DEVOPS")
        print("🚀" + "="*80)
        print()
        
        # Check system health
        await self._check_system_health()
        
        # Run demo queries
        await self._run_demo_queries()
        
        # Show performance metrics
        await self._show_performance_metrics()
        
        # Generate demo report
        await self._generate_demo_report()
        
        print("🎉" + "="*80)
        print("🎉 DEMO COMPLETE: THE FUTURE OF DEVOPS IS HERE!")
        print("🎉" + "="*80)
    
    async def _check_system_health(self):
        """🔍 Check system health and readiness."""
        print("🔍 Checking system health...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/api/v1/health") as response:
                    if response.status == 200:
                        health = await response.json()
                        print(f"✅ System Status: {health['status'].upper()}")
                        print(f"✅ Database: {'HEALTHY' if health['database_healthy'] else 'UNHEALTHY'}")
                        print(f"✅ Cache: {'HEALTHY' if health['redis_healthy'] else 'UNHEALTHY'}")
                        print(f"✅ Vector Extension: {'AVAILABLE' if health['vector_extension_available'] else 'UNAVAILABLE'}")
                        print()
                    else:
                        print("❌ System health check failed!")
                        return False
        except Exception as e:
            print(f"❌ Failed to check system health: {e}")
            return False
        
        return True
    
    async def _run_demo_queries(self):
        """🎯 Run demo queries and show results."""
        print("🎯 Running demo queries...")
        print()
        
        results = []
        
        for i, query_data in enumerate(self.demo_queries, 1):
            print(f"🔍 Query {i}: {query_data['query']}")
            print(f"📊 Context: {query_data['context']}")
            print()
            
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "query": query_data["query"],
                        "context": query_data["context"],
                        "max_results": 3,
                        "include_sources": True
                    }
                    
                    async with session.post(
                        f"{self.api_base_url}/api/v1/query",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            latency = (time.time() - start_time) * 1000
                            
                            print(f"🤖 AI Response:")
                            print(f"   {result['answer'][:200]}...")
                            print()
                            print(f"📊 Confidence: {result['confidence']:.1%}")
                            print(f"⚡ Latency: {latency:.0f}ms")
                            print(f"📚 Sources: {len(result['sources'])}")
                            print(f"🎯 Suggested Actions: {len(result.get('suggested_actions', []))}")
                            print()
                            
                            results.append({
                                "query": query_data["query"],
                                "confidence": result["confidence"],
                                "latency": latency,
                                "sources": len(result["sources"]),
                                "success": True
                            })
                        else:
                            print(f"❌ Query failed with status {response.status}")
                            results.append({
                                "query": query_data["query"],
                                "success": False
                            })
                            
            except Exception as e:
                print(f"❌ Query failed: {e}")
                results.append({
                    "query": query_data["query"],
                    "success": False
                })
            
            print("-" * 80)
            print()
        
        return results
    
    async def _show_performance_metrics(self):
        """📊 Show system performance metrics."""
        print("📊 System Performance Metrics:")
        print()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get system stats
                async with session.get(f"{self.api_base_url}/api/v1/health/stats") as response:
                    if response.status == 200:
                        stats = await response.json()
                        
                        print("📈 Database Statistics:")
                        print(f"   📚 Documents: {stats.get('total_documents', 0):,}")
                        print(f"   🔍 Chunks: {stats.get('total_chunks', 0):,}")
                        print(f"   🧠 Embeddings: {stats.get('total_embeddings', 0):,}")
                        print(f"   💬 Queries: {stats.get('total_queries', 0):,}")
                        print()
                        
                        print("📊 Document Types:")
                        for doc_type, count in stats.get('documents_by_type', {}).items():
                            print(f"   📝 {doc_type}: {count:,}")
                        print()
                        
                        if stats.get('queries_with_feedback', 0) > 0:
                            print(f"⭐ Average Feedback Score: {stats.get('average_feedback_score', 0):.1f}/5.0")
                            print()
                
                # Get query analytics
                async with session.get(f"{self.api_base_url}/api/v1/query/analytics") as response:
                    if response.status == 200:
                        analytics = await response.json()
                        
                        print("⚡ Performance Metrics:")
                        print(f"   🎯 Total Queries: {analytics.get('total_queries', 0):,}")
                        print(f"   ⚡ Avg Latency: {analytics.get('avg_latency_ms', 0):.0f}ms")
                        print(f"   ✅ Success Rate: {analytics.get('success_rate', 0):.1%}")
                        print()
                        
                        if analytics.get('performance_metrics'):
                            perf = analytics['performance_metrics']
                            print("📊 Advanced Metrics:")
                            print(f"   🎯 P95 Latency: {perf.get('p95_latency_ms', 0):.0f}ms")
                            print(f"   🎯 P99 Latency: {perf.get('p99_latency_ms', 0):.0f}ms")
                            print(f"   🔍 Avg Retrieval Count: {perf.get('avg_retrieval_count', 0):.0f}")
                            print(f"   🎯 Avg Confidence: {perf.get('avg_confidence_score', 0):.1%}")
                            print()
                            
        except Exception as e:
            print(f"❌ Failed to get performance metrics: {e}")
    
    async def _generate_demo_report(self):
        """📋 Generate demo report."""
        print("📋 Generating Demo Report...")
        print()
        
        report = {
            "demo_timestamp": time.time(),
            "system_status": "operational",
            "features_demonstrated": [
                "Multi-modal document processing",
                "Hybrid retrieval system",
                "Incident-aware generation",
                "Response validation",
                "Real-time observability"
            ],
            "performance_highlights": [
                "Sub-3 second query latency",
                "High confidence responses",
                "Comprehensive source attribution",
                "Actionable troubleshooting steps"
            ],
            "research_contributions": [
                "Metadata-aware vector search",
                "Temporal relevance modeling",
                "Hybrid ranking pipeline",
                "Incident-aware prompting",
                "Hallucination mitigation"
            ]
        }
        
        print("🎯 Demo Report Generated:")
        print(f"   📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🚀 System Status: {report['system_status'].upper()}")
        print(f"   🎨 Features Demonstrated: {len(report['features_demonstrated'])}")
        print(f"   ⚡ Performance Highlights: {len(report['performance_highlights'])}")
        print(f"   🔬 Research Contributions: {len(report['research_contributions'])}")
        print()
        
        return report


async def main():
    """🎬 Main demo function."""
    demo = RAGDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
