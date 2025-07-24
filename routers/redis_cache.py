"""
🔧 Redis Cache Manager for Enterprise HR Intelligence System
Production-Ready Redis Integration with Fallback Support
"""

import aioredis
import json
import pickle
import asyncio
import hashlib
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import structlog
from collections import defaultdict

logger = structlog.get_logger("redis_cache")

class RedisConfig:
    """📋 Redis configuration"""
    def __init__(self):
        self.url = "redis://localhost:6379/0"
        self.max_connections = 20
        self.socket_timeout = 5
        self.socket_connect_timeout = 5
        self.retry_on_timeout = True
        self.health_check_interval = 30
        self.default_ttl = 300  # 5 minutes

class FallbackCache:
    """💾 In-memory fallback cache when Redis is unavailable"""
    
    def __init__(self):
        self.cache = {}
        self.ttl_cache = {}
        self.stats = defaultdict(int)
    
    def get(self, key: str) -> Optional[Any]:
        """Get from memory cache"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
            
        # Check TTL
        if key in self.ttl_cache:
            if datetime.now() > self.ttl_cache[key]:
                self._delete(key)
                self.stats["misses"] += 1
                return None
        
        self.stats["hits"] += 1
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set in memory cache"""
        try:
            self.cache[key] = value
            self.ttl_cache[key] = datetime.now() + timedelta(seconds=ttl)
            self.stats["sets"] += 1
            return True
        except Exception:
            self.stats["errors"] += 1
            return False
    
    def _delete(self, key: str):
        """Internal delete method"""
        self.cache.pop(key, None)
        self.ttl_cache.pop(key, None)
    
    def delete(self, key: str) -> bool:
        """Delete from memory cache"""
        if key in self.cache:
            self._delete(key)
            self.stats["deletes"] += 1
            return True
        return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        keys_to_delete = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_delete:
            self._delete(key)
        self.stats["deletes"] += len(keys_to_delete)
        return len(keys_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_ops = sum(self.stats.values())
        hit_rate = self.stats["hits"] / max(total_ops, 1)
        
        return {
            "type": "fallback_memory",
            "total_keys": len(self.cache),
            "hit_rate": round(hit_rate, 3),
            "statistics": dict(self.stats)
        }

class EnterpriseRedisCache:
    """🧠 Enterprise Redis cache with intelligent fallback"""
    
    def __init__(self, config: RedisConfig = None):
        self.config = config or RedisConfig()
        self.redis: Optional[aioredis.Redis] = None
        self.fallback_cache = FallbackCache()
        self.redis_available = False
        self.key_prefix = "hr_intelligence:"
        self.stats = defaultdict(int)
        self._connection_lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """🚀 Initialize Redis connection"""
        try:
            async with self._connection_lock:
                if self.redis is None:
                    self.redis = aioredis.from_url(
                        self.config.url,
                        max_connections=self.config.max_connections,
                        socket_timeout=self.config.socket_timeout,
                        socket_connect_timeout=self.config.socket_connect_timeout,
                        retry_on_timeout=self.config.retry_on_timeout,
                        encoding="utf-8",
                        decode_responses=False  # Handle binary data
                    )
                    
                    # Test connection
                    await asyncio.wait_for(self.redis.ping(), timeout=5.0)
                    self.redis_available = True
                    logger.info("Redis connection established successfully")
                    return True
                    
        except Exception as e:
            logger.warning("Redis initialization failed, using fallback cache", error=str(e))
            self.redis_available = False
            return False
    
    async def health_check(self) -> bool:
        """🏥 Check Redis health"""
        if not self.redis:
            return False
            
        try:
            await asyncio.wait_for(self.redis.ping(), timeout=2.0)
            if not self.redis_available:
                logger.info("Redis connection restored")
                self.redis_available = True
            return True
        except Exception as e:
            if self.redis_available:
                logger.warning("Redis connection lost, switching to fallback", error=str(e))
                self.redis_available = False
            return False
    
    def _make_key(self, key: str) -> str:
        """🔑 Generate cache key with prefix"""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """📥 Get value from cache (Redis or fallback)"""
        cache_key = self._make_key(key)
        
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                data = await self.redis.get(cache_key)
                if data:
                    # Try to deserialize as JSON first, then pickle
                    try:
                        result = json.loads(data.decode('utf-8'))
                        self.stats["redis_hits"] += 1
                        return result
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        try:
                            result = pickle.loads(data)
                            self.stats["redis_hits"] += 1
                            return result
                        except Exception:
                            logger.warning("Failed to deserialize cached data", key=key)
                
                self.stats["redis_misses"] += 1
                return None
                
            except Exception as e:
                logger.warning("Redis get failed, trying fallback", key=key, error=str(e))
                await self.health_check()  # Update Redis status
        
        # Use fallback cache
        result = self.fallback_cache.get(cache_key)
        if result is not None:
            self.stats["fallback_hits"] += 1
        else:
            self.stats["fallback_misses"] += 1
        return result
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """📤 Set value in cache"""
        cache_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl
        
        # Try Redis first if available
        if self.redis_available and self.redis:
            try:
                # Try JSON serialization first
                try:
                    serialized = json.dumps(value, default=str).encode('utf-8')
                except (TypeError, ValueError):
                    # Fall back to pickle for complex objects
                    serialized = pickle.dumps(value)
                
                result = await self.redis.setex(cache_key, ttl, serialized)
                self.stats["redis_sets"] += 1
                
                # Also set in fallback for redundancy
                self.fallback_cache.set(cache_key, value, ttl)
                
                return bool(result)
                
            except Exception as e:
                logger.warning("Redis set failed, using fallback only", key=key, error=str(e))
                await self.health_check()
        
        # Use fallback cache
        result = self.fallback_cache.set(cache_key, value, ttl)
        if result:
            self.stats["fallback_sets"] += 1
        return result
    
    async def delete(self, key: str) -> bool:
        """🗑️ Delete from cache"""
        cache_key = self._make_key(key)
        redis_result = False
        fallback_result = False
        
        # Delete from Redis
        if self.redis_available and self.redis:
            try:
                redis_result = bool(await self.redis.delete(cache_key))
                self.stats["redis_deletes"] += 1
            except Exception as e:
                logger.warning("Redis delete failed", key=key, error=str(e))
                await self.health_check()
        
        # Delete from fallback
        fallback_result = self.fallback_cache.delete(cache_key)
        if fallback_result:
            self.stats["fallback_deletes"] += 1
        
        return redis_result or fallback_result
    
    async def delete_pattern(self, pattern: str) -> int:
        """🗑️ Delete keys matching pattern"""
        pattern_key = self._make_key(pattern)
        total_deleted = 0
        
        # Delete from Redis
        if self.redis_available and self.redis:
            try:
                keys = await self.redis.keys(f"{pattern_key}*")
                if keys:
                    deleted = await self.redis.delete(*keys)
                    total_deleted += deleted
                    self.stats["redis_pattern_deletes"] += 1
            except Exception as e:
                logger.warning("Redis pattern delete failed", pattern=pattern, error=str(e))
                await self.health_check()
        
        # Delete from fallback
        fallback_deleted = self.fallback_cache.delete_pattern(pattern_key)
        total_deleted += fallback_deleted
        
        return total_deleted
    
    async def get_stats(self) -> Dict[str, Any]:
        """📊 Get comprehensive cache statistics"""
        redis_stats = {}
        if self.redis and self.redis_available:
            try:
                info = await self.redis.info()
                redis_stats = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
            except Exception:
                redis_stats = {"error": "Unable to fetch Redis stats"}
        
        fallback_stats = self.fallback_cache.get_stats()
        
        # Calculate hit rates
        total_redis_ops = self.stats["redis_hits"] + self.stats["redis_misses"]
        total_fallback_ops = self.stats["fallback_hits"] + self.stats["fallback_misses"]
        
        redis_hit_rate = self.stats["redis_hits"] / max(total_redis_ops, 1)
        fallback_hit_rate = self.stats["fallback_hits"] / max(total_fallback_ops, 1)
        
        return {
            "redis_available": self.redis_available,
            "redis_stats": redis_stats,
            "redis_hit_rate": round(redis_hit_rate, 3),
            "fallback_stats": fallback_stats,
            "fallback_hit_rate": round(fallback_hit_rate, 3),
            "operation_stats": dict(self.stats),
            "total_operations": sum(self.stats.values())
        }
    
    async def close(self):
        """🔚 Close Redis connection"""
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            self.redis_available = False
            logger.info("Redis connection closed")