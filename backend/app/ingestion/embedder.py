"""
Multi-provider embedding system with OpenAI and sentence-transformers fallback.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
import tiktoken
from dataclasses import dataclass
import asyncio
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor
import time

from ..core.config import settings
from ..core.cache import embedding_cache

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embeddings: List[List[float]]
    model: str
    provider: str
    token_count: int
    latency_ms: float
    cached: bool = False


class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider_name = "base"
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        raise NotImplementedError
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimension size for this model."""
        raise NotImplementedError
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model_name: str = "text-embedding-3-large", api_key: Optional[str] = None):
        super().__init__(model_name)
        self.provider_name = "openai"
        self.api_key = api_key or settings.openai_api_key
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Model configurations
        self.model_configs = {
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191}
        }
        
        self.config = self.model_configs.get(model_name, {"dimensions": 1536, "max_tokens": 8191})
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        start_time = time.time()
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = embedding_cache.get_embedding(text, self.model_name)
            if cached_embedding:
                cached_embeddings.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            try:
                response = await self._call_openai_api(uncached_texts)
                new_embeddings = [item.embedding for item in response.data]
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    embedding_cache.set_embedding(text, embedding, self.model_name)
                    
            except Exception as e:
                logger.error(f"OpenAI embedding failed: {e}")
                raise
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding
        
        latency_ms = (time.time() - start_time) * 1000
        token_count = sum(self.estimate_tokens(text) for text in texts)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model_name,
            provider=self.provider_name,
            token_count=token_count,
            latency_ms=latency_ms,
            cached=len(cached_embeddings) == len(texts)
        )
    
    async def _call_openai_api(self, texts: List[str]) -> Any:
        """Call OpenAI API with proper error handling."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )
            return response
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions for this model."""
        return self.config["dimensions"]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback estimation
            return len(text.split()) * 1.3


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence-transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.provider_name = "sentence_transformers"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings using sentence-transformers."""
        start_time = time.time()
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_embedding = embedding_cache.get_embedding(text, self.model_name)
            if cached_embedding:
                cached_embeddings.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    new_embeddings = await loop.run_in_executor(
                        executor, self.model.encode, uncached_texts
                    )
                
                # Convert to list of lists
                new_embeddings = [embedding.tolist() for embedding in new_embeddings]
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    embedding_cache.set_embedding(text, embedding, self.model_name)
                    
            except Exception as e:
                logger.error(f"Sentence transformer embedding failed: {e}")
                raise
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding
        
        for i, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = embedding
        
        latency_ms = (time.time() - start_time) * 1000
        token_count = sum(self.estimate_tokens(text) for text in texts)
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model_name,
            provider=self.provider_name,
            token_count=token_count,
            latency_ms=latency_ms,
            cached=len(cached_embeddings) == len(texts)
        )
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions for this model."""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for sentence transformers."""
        # Rough estimation based on word count
        return len(text.split()) * 1.3


class MultiProviderEmbedder:
    """
    Multi-provider embedding system with fallback logic.
    Supports OpenAI and sentence-transformers with caching.
    """
    
    def __init__(self, primary_provider: str = "openai", fallback_provider: str = "sentence_transformers"):
        self.primary_provider_name = primary_provider
        self.fallback_provider_name = fallback_provider
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize embedding providers."""
        # Initialize OpenAI provider
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIEmbeddingProvider(
                    model_name=settings.embedding_model,
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI embedding provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize sentence-transformers provider
        try:
            self.providers["sentence_transformers"] = SentenceTransformerProvider(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Sentence transformers provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize sentence transformers provider: {e}")
    
    async def embed_texts(self, texts: List[str], provider: Optional[str] = None) -> EmbeddingResult:
        """
        Generate embeddings with automatic fallback.
        
        Args:
            texts: List of texts to embed
            provider: Specific provider to use (optional)
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model="none",
                provider="none",
                token_count=0,
                latency_ms=0
            )
        
        # Try primary provider first
        if provider is None:
            provider = self.primary_provider_name
        
        if provider in self.providers:
            try:
                result = await self.providers[provider].embed_texts(texts)
                logger.info(f"Generated embeddings using {provider} provider")
                return result
            except Exception as e:
                logger.warning(f"Primary provider {provider} failed: {e}")
        
        # Try fallback provider
        fallback_provider = self.fallback_provider_name
        if fallback_provider in self.providers and fallback_provider != provider:
            try:
                result = await self.providers[fallback_provider].embed_texts(texts)
                logger.info(f"Generated embeddings using fallback provider {fallback_provider}")
                return result
            except Exception as e:
                logger.error(f"Fallback provider {fallback_provider} failed: {e}")
        
        # If all providers fail, raise error
        raise RuntimeError("All embedding providers failed")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        if provider_name not in self.providers:
            return None
        
        provider = self.providers[provider_name]
        return {
            "name": provider_name,
            "model": provider.model_name,
            "dimensions": provider.get_embedding_dimensions(),
            "provider": provider.provider_name
        }
    
    async def batch_embed(self, texts: List[str], batch_size: int = 100) -> EmbeddingResult:
        """
        Generate embeddings in batches for large text collections.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of each batch
            
        Returns:
            Combined EmbeddingResult for all texts
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model="none",
                provider="none",
                token_count=0,
                latency_ms=0
            )
        
        all_embeddings = []
        total_token_count = 0
        total_latency = 0
        used_provider = None
        used_model = None
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                result = await self.embed_texts(batch)
                all_embeddings.extend(result.embeddings)
                total_token_count += result.token_count
                total_latency += result.latency_ms
                used_provider = result.provider
                used_model = result.model
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {i//batch_size + 1}: {e}")
                raise
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=used_model or "unknown",
            provider=used_provider or "unknown",
            token_count=total_token_count,
            latency_ms=total_latency
        )
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = await self.embed_texts([text])
        return result.embeddings[0] if result.embeddings else []
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions from the primary provider."""
        if self.primary_provider_name in self.providers:
            return self.providers[self.primary_provider_name].get_embedding_dimensions()
        elif self.fallback_provider_name in self.providers:
            return self.providers[self.fallback_provider_name].get_embedding_dimensions()
        else:
            return 384  # Default fallback


# Global embedder instance
embedder = MultiProviderEmbedder()


# Utility functions for research analysis
async def analyze_embedding_quality(embeddings: List[List[float]]) -> Dict[str, Any]:
    """Analyze embedding quality for research purposes."""
    if not embeddings:
        return {"quality_score": 0.0, "analysis": "No embeddings provided"}
    
    embeddings_array = np.array(embeddings)
    
    # Calculate various quality metrics
    mean_norm = np.mean(np.linalg.norm(embeddings_array, axis=1))
    std_norm = np.std(np.linalg.norm(embeddings_array, axis=1))
    
    # Calculate pairwise cosine similarities
    normalized_embeddings = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    cosine_similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Remove diagonal (self-similarity)
    mask = np.ones_like(cosine_similarities, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diagonal_similarities = cosine_similarities[mask]
    
    mean_similarity = np.mean(off_diagonal_similarities)
    std_similarity = np.std(off_diagonal_similarities)
    
    # Calculate quality score (lower similarity variance is better)
    quality_score = 1.0 - std_similarity
    
    return {
        "quality_score": float(quality_score),
        "mean_norm": float(mean_norm),
        "std_norm": float(std_norm),
        "mean_similarity": float(mean_similarity),
        "std_similarity": float(std_similarity),
        "embedding_count": len(embeddings),
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "analysis": "Good quality" if quality_score > 0.7 else "Poor quality" if quality_score < 0.3 else "Medium quality"
    }


async def compare_embedding_providers(texts: List[str]) -> Dict[str, Any]:
    """Compare different embedding providers for research analysis."""
    results = {}
    
    for provider_name in embedder.get_available_providers():
        try:
            start_time = time.time()
            result = await embedder.embed_texts(texts, provider=provider_name)
            latency = time.time() - start_time
            
            quality = await analyze_embedding_quality(result.embeddings)
            
            results[provider_name] = {
                "latency_ms": result.latency_ms,
                "token_count": result.token_count,
                "cached": result.cached,
                "quality_score": quality["quality_score"],
                "dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                "provider_info": embedder.get_provider_info(provider_name)
            }
            
        except Exception as e:
            results[provider_name] = {
                "error": str(e),
                "available": False
            }
    
    return results
