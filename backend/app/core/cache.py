"""
Redis cache management for the research system.
"""

import json
import logging
from typing import Any, Optional, Dict, List
import redis
from redis.exceptions import RedisError
import pickle
import hashlib

from .config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager for research operations.
    Handles caching of embeddings, query results, and other research data.
    """
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=False,  # We'll handle encoding manually
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate a cache key with prefix and identifier."""
        # Create a hash of the identifier to ensure key length limits
        hash_obj = hashlib.md5(identifier.encode())
        hash_hex = hash_obj.hexdigest()
        return f"{prefix}:{hash_hex}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage in Redis."""
        try:
            # Try JSON serialization first (for simple data types)
            if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(data).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize data: {e}")
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis."""
        try:
            # Try JSON deserialization first
            decoded = data.decode('utf-8')
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                # Fall back to pickle
                return pickle.loads(data)
            except Exception as e:
                logger.warning(f"Failed to deserialize data: {e}")
                return None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not self.redis_client:
            return default
        
        try:
            data = self.redis_client.get(key)
            if data is None:
                return default
            return self._deserialize_data(data)
        except RedisError as e:
            logger.error(f"Failed to get key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if not self.redis_client:
            return False
        
        try:
            serialized_data = self._serialize_data(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_data)
            else:
                return self.redis_client.set(key, serialized_data)
        except RedisError as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except RedisError as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except RedisError as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for a key."""
        if not self.redis_client:
            return -1
        
        try:
            return self.redis_client.ttl(key)
        except RedisError as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            return -1
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except RedisError as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0


class EmbeddingCache:
    """
    Specialized cache for embeddings with research-specific optimizations.
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.embedding_prefix = "emb"
        self.embedding_ttl = settings.redis_ttl
    
    def get_embedding(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = self._get_embedding_key(text, model)
        return self.cache.get(key)
    
    def set_embedding(self, text: str, embedding: List[float], model: str = "default") -> bool:
        """Cache embedding for text."""
        key = self._get_embedding_key(text, model)
        return self.cache.set(key, embedding, self.embedding_ttl)
    
    def _get_embedding_key(self, text: str, model: str) -> str:
        """Generate embedding cache key."""
        identifier = f"{model}:{text}"
        return self.cache._generate_key(self.embedding_prefix, identifier)
    
    def clear_embeddings(self, model: str = None) -> int:
        """Clear cached embeddings."""
        if model:
            pattern = f"{self.embedding_prefix}:{model}:*"
        else:
            pattern = f"{self.embedding_prefix}:*"
        return self.cache.clear_pattern(pattern)


class QueryCache:
    """
    Specialized cache for query results and research data.
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.query_prefix = "query"
        self.result_prefix = "result"
        self.query_ttl = 3600  # 1 hour for query results
    
    def get_query_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        key = f"{self.result_prefix}:{query_hash}"
        return self.cache.get(key)
    
    def set_query_result(self, query_hash: str, result: Dict[str, Any]) -> bool:
        """Cache query result."""
        key = f"{self.result_prefix}:{query_hash}"
        return self.cache.set(key, result, self.query_ttl)
    
    def get_retrieval_cache(self, query: str, filters: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval results."""
        cache_key = self._get_retrieval_key(query, filters)
        return self.cache.get(cache_key)
    
    def set_retrieval_cache(self, query: str, filters: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
        """Cache retrieval results."""
        cache_key = self._get_retrieval_key(query, filters)
        return self.cache.set(cache_key, results, self.query_ttl)
    
    def _get_retrieval_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Generate retrieval cache key."""
        # Create a hash of query and filters
        content = f"{query}:{json.dumps(filters, sort_keys=True)}"
        identifier = hashlib.md5(content.encode()).hexdigest()
        return self.cache._generate_key(self.query_prefix, identifier)
    
    def clear_query_cache(self) -> int:
        """Clear all query-related cache."""
        query_pattern = f"{self.query_prefix}:*"
        result_pattern = f"{self.result_prefix}:*"
        cleared_queries = self.cache.clear_pattern(query_pattern)
        cleared_results = self.cache.clear_pattern(result_pattern)
        return cleared_queries + cleared_results


class ResearchCache:
    """
    Research-specific cache for experiment data and analytics.
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.experiment_prefix = "exp"
        self.metrics_prefix = "metrics"
        self.analytics_ttl = 86400  # 24 hours for analytics data
    
    def get_experiment_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get cached experiment data."""
        key = f"{self.experiment_prefix}:{experiment_id}"
        return self.cache.get(key)
    
    def set_experiment_data(self, experiment_id: str, data: Dict[str, Any]) -> bool:
        """Cache experiment data."""
        key = f"{self.experiment_prefix}:{experiment_id}"
        return self.cache.set(key, data, self.analytics_ttl)
    
    def get_metrics(self, metric_name: str, time_range: str = "daily") -> Optional[List[Dict[str, Any]]]:
        """Get cached metrics data."""
        key = f"{self.metrics_prefix}:{metric_name}:{time_range}"
        return self.cache.get(key)
    
    def set_metrics(self, metric_name: str, data: List[Dict[str, Any]], time_range: str = "daily") -> bool:
        """Cache metrics data."""
        key = f"{self.metrics_prefix}:{metric_name}:{time_range}"
        return self.cache.set(key, data, self.analytics_ttl)
    
    def clear_research_cache(self) -> int:
        """Clear all research-related cache."""
        exp_pattern = f"{self.experiment_prefix}:*"
        metrics_pattern = f"{self.metrics_prefix}:*"
        cleared_exp = self.cache.clear_pattern(exp_pattern)
        cleared_metrics = self.cache.clear_pattern(metrics_pattern)
        return cleared_exp + cleared_metrics


# Global cache instances
cache_manager = CacheManager()
embedding_cache = EmbeddingCache(cache_manager)
query_cache = QueryCache(cache_manager)
research_cache = ResearchCache(cache_manager)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    if not cache_manager.redis_client:
        return {"status": "disconnected", "stats": {}}
    
    try:
        info = cache_manager.redis_client.info()
        return {
            "status": "connected",
            "stats": {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            }
        }
    except RedisError as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"status": "error", "stats": {}}
