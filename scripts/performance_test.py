#!/usr/bin/env python3
"""
⚡ Performance Testing Script for the Most Advanced RAG System
"""

import asyncio
import aiohttp
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Any
import csv

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """⚡ Comprehensive performance testing for the RAG system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_queries = [
            {
                "query": "🚨 API Gateway is returning 503 errors, what should I do?",
                "context": {"service": "api-gateway", "severity": "critical"},
                "expected_latency": 3000,  # 3 seconds
                "expected_confidence": 0.8
            },
            {
                "query": "🐌 Database queries are running slow, how can I optimize?",
                "context": {"service": "database", "severity": "high"},
                "expected_latency": 2500,  # 2.5 seconds
                "expected_confidence": 0.85
            },
            {
                "query": "🔐 Authentication is failing, what's wrong?",
                "context": {"service": "auth-service", "component": "authentication"},
                "expected_latency": 2000,  # 2 seconds
                "expected_confidence": 0.9
            },
            {
                "query": "📊 How to monitor service health?",
                "context": {"service": "monitoring", "incident_type": "monitoring"},
                "expected_latency": 1500,  # 1.5 seconds
                "expected_confidence": 0.95
            },
            {
                "query": "🚀 How to rollback a failed deployment?",
                "context": {"service": "deployment", "incident_type": "deployment_issues"},
                "expected_latency": 2000,  # 2 seconds
                "expected_confidence": 0.88
            }
        ]
    
    async def run_performance_test(self, duration_minutes: int = 10):
        """⚡ Run comprehensive performance tests."""
        print("⚡" + "="*80)
        print("⚡ PERFORMANCE TESTING: THE MOST ADVANCED RAG SYSTEM")
        print("⚡" + "="*80)
        print()
        
        # Test configuration
        test_duration = duration_minutes * 60  # Convert to seconds
        start_time = time.time()
        
        # Initialize test results
        results = {
            "test_metadata": {
                "start_time": datetime.now().isoformat(),
                "duration_minutes": duration_minutes,
                "api_base_url": self.api_base_url
            },
            "latency_metrics": [],
            "accuracy_metrics": [],
            "throughput_metrics": [],
            "error_metrics": [],
            "system_health": {}
        }
        
        # Run tests
        await self._run_latency_tests(results)
        await self._run_accuracy_tests(results)
        await self._run_throughput_tests(results, test_duration)
        await self._run_load_tests(results)
        await self._check_system_health(results)
        
        # Generate report
        await self._generate_performance_report(results)
        
        return results
    
    async def _run_latency_tests(self, results: Dict[str, Any]):
        """🎯 Test query latency performance."""
        print("🎯 Testing query latency performance...")
        
        latency_results = []
        
        for i, query_data in enumerate(self.test_queries):
            print(f"   Query {i+1}: {query_data['query'][:50]}...")
            
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
                            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                            
                            latency_results.append({
                                "query_id": i+1,
                                "latency_ms": latency,
                                "expected_latency_ms": query_data["expected_latency"],
                                "confidence": result.get("confidence", 0),
                                "sources_count": len(result.get("sources", [])),
                                "success": True
                            })
                            
                            print(f"      ✅ Latency: {latency:.0f}ms (Expected: {query_data['expected_latency']}ms)")
                        else:
                            print(f"      ❌ Failed with status {response.status}")
                            latency_results.append({
                                "query_id": i+1,
                                "latency_ms": 0,
                                "success": False
                            })
                            
            except Exception as e:
                print(f"      ❌ Error: {e}")
                latency_results.append({
                    "query_id": i+1,
                    "latency_ms": 0,
                    "success": False
                })
        
        results["latency_metrics"] = latency_results
        
        # Calculate latency statistics
        successful_latencies = [r["latency_ms"] for r in latency_results if r["success"]]
        if successful_latencies:
            results["latency_summary"] = {
                "avg_latency_ms": statistics.mean(successful_latencies),
                "median_latency_ms": statistics.median(successful_latencies),
                "p95_latency_ms": self._calculate_percentile(successful_latencies, 95),
                "p99_latency_ms": self._calculate_percentile(successful_latencies, 99),
                "min_latency_ms": min(successful_latencies),
                "max_latency_ms": max(successful_latencies),
                "success_rate": len(successful_latencies) / len(latency_results)
            }
        
        print()
    
    async def _run_accuracy_tests(self, results: Dict[str, Any]):
        """🎯 Test response accuracy and quality."""
        print("🎯 Testing response accuracy and quality...")
        
        accuracy_results = []
        
        for i, query_data in enumerate(self.test_queries):
            print(f"   Query {i+1}: {query_data['query'][:50]}...")
            
            try:
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
                            
                            # Evaluate response quality
                            confidence = result.get("confidence", 0)
                            sources = result.get("sources", [])
                            suggested_actions = result.get("suggested_actions", [])
                            
                            # Calculate accuracy score
                            accuracy_score = self._calculate_accuracy_score(
                                confidence, len(sources), len(suggested_actions)
                            )
                            
                            accuracy_results.append({
                                "query_id": i+1,
                                "confidence": confidence,
                                "expected_confidence": query_data["expected_confidence"],
                                "sources_count": len(sources),
                                "suggested_actions_count": len(suggested_actions),
                                "accuracy_score": accuracy_score,
                                "meets_expectations": confidence >= query_data["expected_confidence"]
                            })
                            
                            print(f"      ✅ Confidence: {confidence:.1%} (Expected: {query_data['expected_confidence']:.1%})")
                            print(f"      📚 Sources: {len(sources)}")
                            print(f"      🎯 Actions: {len(suggested_actions)}")
                        else:
                            print(f"      ❌ Failed with status {response.status}")
                            
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        results["accuracy_metrics"] = accuracy_results
        
        # Calculate accuracy statistics
        if accuracy_results:
            confidences = [r["confidence"] for r in accuracy_results]
            accuracy_scores = [r["accuracy_score"] for r in accuracy_results]
            meets_expectations = [r["meets_expectations"] for r in accuracy_results]
            
            results["accuracy_summary"] = {
                "avg_confidence": statistics.mean(confidences),
                "avg_accuracy_score": statistics.mean(accuracy_scores),
                "expectations_met_rate": sum(meets_expectations) / len(meets_expectations),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences)
            }
        
        print()
    
    async def _run_throughput_tests(self, results: Dict[str, Any], test_duration: int):
        """⚡ Test system throughput and concurrent performance."""
        print("⚡ Testing system throughput and concurrent performance...")
        
        # Test concurrent queries
        concurrent_queries = [5, 10, 20, 50]  # Different concurrency levels
        throughput_results = []
        
        for concurrency in concurrent_queries:
            print(f"   Testing {concurrency} concurrent queries...")
            
            start_time = time.time()
            successful_queries = 0
            total_latency = 0
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrency):
                task = self._run_single_query(i, results)
                tasks.append(task)
            
            # Wait for all tasks to complete
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for task_result in task_results:
                if isinstance(task_result, dict) and task_result.get("success"):
                    successful_queries += 1
                    total_latency += task_result.get("latency_ms", 0)
            
            end_time = time.time()
            test_duration_actual = end_time - start_time
            
            throughput_results.append({
                "concurrency": concurrency,
                "successful_queries": successful_queries,
                "total_queries": concurrency,
                "success_rate": successful_queries / concurrency,
                "avg_latency_ms": total_latency / successful_queries if successful_queries > 0 else 0,
                "queries_per_second": successful_queries / test_duration_actual,
                "test_duration_seconds": test_duration_actual
            })
            
            print(f"      ✅ Success Rate: {successful_queries/concurrency:.1%}")
            print(f"      ⚡ QPS: {successful_queries/test_duration_actual:.1f}")
            print(f"      🎯 Avg Latency: {total_latency/successful_queries:.0f}ms")
        
        results["throughput_metrics"] = throughput_results
        print()
    
    async def _run_load_tests(self, results: Dict[str, Any]):
        """🔥 Test system under sustained load."""
        print("🔥 Testing system under sustained load...")
        
        # Run continuous queries for 5 minutes
        load_test_duration = 300  # 5 minutes
        start_time = time.time()
        query_count = 0
        successful_queries = 0
        error_count = 0
        
        print(f"   Running load test for {load_test_duration} seconds...")
        
        while time.time() - start_time < load_test_duration:
            try:
                async with aiohttp.ClientSession() as session:
                    # Random query selection
                    import random
                    query_data = random.choice(self.test_queries)
                    
                    payload = {
                        "query": query_data["query"],
                        "context": query_data["context"],
                        "max_results": 3
                    }
                    
                    query_start = time.time()
                    async with session.post(
                        f"{self.api_base_url}/api/v1/query",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        query_latency = (time.time() - query_start) * 1000
                        
                        if response.status == 200:
                            successful_queries += 1
                        else:
                            error_count += 1
                        
                        query_count += 1
                        
                        # Log progress every 30 seconds
                        if query_count % 10 == 0:
                            elapsed = time.time() - start_time
                            print(f"      Progress: {query_count} queries in {elapsed:.0f}s")
                
            except Exception as e:
                error_count += 1
                query_count += 1
        
        total_duration = time.time() - start_time
        
        results["load_test_results"] = {
            "total_queries": query_count,
            "successful_queries": successful_queries,
            "error_count": error_count,
            "success_rate": successful_queries / query_count if query_count > 0 else 0,
            "queries_per_second": query_count / total_duration,
            "test_duration_seconds": total_duration
        }
        
        print(f"   ✅ Load Test Complete:")
        print(f"      📊 Total Queries: {query_count}")
        print(f"      ✅ Successful: {successful_queries}")
        print(f"      ❌ Errors: {error_count}")
        print(f"      📈 Success Rate: {successful_queries/query_count:.1%}")
        print(f"      ⚡ QPS: {query_count/total_duration:.1f}")
        print()
    
    async def _check_system_health(self, results: Dict[str, Any]):
        """🔍 Check system health and resource utilization."""
        print("🔍 Checking system health and resource utilization...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get system health
                async with session.get(f"{self.api_base_url}/api/v1/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results["system_health"] = health_data
                        
                        print(f"   ✅ System Status: {health_data.get('status', 'unknown')}")
                        print(f"   🗄️ Database: {'Healthy' if health_data.get('database_healthy') else 'Unhealthy'}")
                        print(f"   ⚡ Redis: {'Healthy' if health_data.get('redis_healthy') else 'Unhealthy'}")
                        print(f"   🧮 Vector Extension: {'Available' if health_data.get('vector_extension_available') else 'Unavailable'}")
                
                # Get system stats
                async with session.get(f"{self.api_base_url}/api/v1/health/stats") as response:
                    if response.status == 200:
                        stats_data = await response.json()
                        results["system_stats"] = stats_data
                        
                        print(f"   📚 Documents: {stats_data.get('total_documents', 0):,}")
                        print(f"   🔍 Chunks: {stats_data.get('total_chunks', 0):,}")
                        print(f"   🧠 Embeddings: {stats_data.get('total_embeddings', 0):,}")
                        print(f"   💬 Queries: {stats_data.get('total_queries', 0):,}")
                        
        except Exception as e:
            print(f"   ❌ Failed to check system health: {e}")
        
        print()
    
    async def _run_single_query(self, query_id: int, results: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Run a single query for throughput testing."""
        try:
            import random
            query_data = random.choice(self.test_queries)
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query_data["query"],
                    "context": query_data["context"],
                    "max_results": 3
                }
                
                async with session.post(
                    f"{self.api_base_url}/api/v1/query",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "query_id": query_id,
                            "success": True,
                            "latency_ms": latency,
                            "confidence": result.get("confidence", 0)
                        }
                    else:
                        return {
                            "query_id": query_id,
                            "success": False,
                            "latency_ms": latency
                        }
                        
        except Exception as e:
            return {
                "query_id": query_id,
                "success": False,
                "error": str(e)
            }
    
    def _calculate_accuracy_score(self, confidence: float, sources_count: int, actions_count: int) -> float:
        """📊 Calculate accuracy score based on multiple factors."""
        # Weighted scoring: confidence (50%), sources (30%), actions (20%)
        confidence_score = confidence * 0.5
        sources_score = min(sources_count / 5, 1.0) * 0.3  # Normalize to 5 sources max
        actions_score = min(actions_count / 3, 1.0) * 0.2  # Normalize to 3 actions max
        
        return confidence_score + sources_score + actions_score
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """📊 Calculate percentile value."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def _generate_performance_report(self, results: Dict[str, Any]):
        """📊 Generate comprehensive performance report."""
        print("📊 Generating comprehensive performance report...")
        
        # Add summary statistics
        results["test_summary"] = {
            "total_tests": len(results["latency_metrics"]),
            "successful_tests": len([r for r in results["latency_metrics"] if r["success"]]),
            "overall_success_rate": len([r for r in results["latency_metrics"] if r["success"]]) / len(results["latency_metrics"]),
            "test_completion_time": datetime.now().isoformat()
        }
        
        # Save results to files
        await self._save_results_to_files(results)
        
        # Display summary
        await self._display_performance_summary(results)
        
        print("🎉" + "="*80)
        print("🎉 PERFORMANCE TESTING COMPLETE!")
        print("🎉" + "="*80)
    
    async def _save_results_to_files(self, results: Dict[str, Any]):
        """💾 Save results to various file formats."""
        # Save JSON report
        with open("performance_report.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV metrics
        with open("performance_metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Query ID", "Latency (ms)", "Confidence", "Sources", "Success"])
            
            for metric in results["latency_metrics"]:
                writer.writerow([
                    metric["query_id"],
                    metric["latency_ms"],
                    metric.get("confidence", 0),
                    metric.get("sources_count", 0),
                    metric["success"]
                ])
        
        # Save benchmark results
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "latency_summary": results.get("latency_summary", {}),
            "accuracy_summary": results.get("accuracy_summary", {}),
            "throughput_metrics": results.get("throughput_metrics", []),
            "load_test_results": results.get("load_test_results", {}),
            "system_health": results.get("system_health", {})
        }
        
        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print("💾 Results saved to:")
        print("   📊 performance_report.json")
        print("   📈 performance_metrics.csv")
        print("   🎯 benchmark_results.json")
    
    async def _display_performance_summary(self, results: Dict[str, Any]):
        """📊 Display performance test summary."""
        print("📊" + "="*80)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("📊" + "="*80)
        print()
        
        # Latency summary
        if "latency_summary" in results:
            latency = results["latency_summary"]
            print("⚡ LATENCY PERFORMANCE:")
            print(f"   🎯 Average: {latency['avg_latency_ms']:.0f}ms")
            print(f"   📊 Median: {latency['median_latency_ms']:.0f}ms")
            print(f"   🎯 P95: {latency['p95_latency_ms']:.0f}ms")
            print(f"   🎯 P99: {latency['p99_latency_ms']:.0f}ms")
            print(f"   ✅ Success Rate: {latency['success_rate']:.1%}")
            print()
        
        # Accuracy summary
        if "accuracy_summary" in results:
            accuracy = results["accuracy_summary"]
            print("🎯 ACCURACY PERFORMANCE:")
            print(f"   🧠 Average Confidence: {accuracy['avg_confidence']:.1%}")
            print(f"   📊 Accuracy Score: {accuracy['avg_accuracy_score']:.1%}")
            print(f"   ✅ Expectations Met: {accuracy['expectations_met_rate']:.1%}")
            print()
        
        # Throughput summary
        if "throughput_metrics" in results:
            print("⚡ THROUGHPUT PERFORMANCE:")
            for metric in results["throughput_metrics"]:
                print(f"   🔄 {metric['concurrency']} concurrent: {metric['queries_per_second']:.1f} QPS")
            print()
        
        # Load test summary
        if "load_test_results" in results:
            load_test = results["load_test_results"]
            print("🔥 LOAD TEST RESULTS:")
            print(f"   📊 Total Queries: {load_test['total_queries']:,}")
            print(f"   ✅ Success Rate: {load_test['success_rate']:.1%}")
            print(f"   ⚡ QPS: {load_test['queries_per_second']:.1f}")
            print()


async def main():
    """⚡ Main performance testing function."""
    parser = argparse.ArgumentParser(description="Performance testing for RAG system")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in minutes")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    tester = PerformanceTester(api_base_url=args.api_url)
    await tester.run_performance_test(duration_minutes=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
