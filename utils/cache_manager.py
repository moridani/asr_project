from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import time
import json
import shutil
import asyncio
from loguru import logger
import torch
import hashlib
import psutil
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import aiofiles
import os

class CacheManager:
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size_gb: float = 10.0,
        ttl_hours: float = 24.0,
        cleanup_interval: float = 1.0,
        memory_threshold: float = 0.9
    ):
        """
        Initialize cache manager with enhanced memory management.
        
        Args:
            cache_dir: Cache directory path
            max_size_gb: Maximum cache size in GB
            ttl_hours: Time-to-live in hours
            cleanup_interval: Cleanup interval in hours
            memory_threshold: Memory usage threshold (0-1)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = int(max_size_gb * 1024 * 1024 * 1024)
        self.ttl = int(ttl_hours * 3600)
        self.cleanup_interval = int(cleanup_interval * 3600)
        self.memory_threshold = memory_threshold
        self.memory_check_interval = 300  # 5 minutes
        
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.lock = Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self._initialize_cache()
        self._start_monitoring_tasks()

    def _initialize_cache(self):
        """Initialize cache with integrity check."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = self._load_metadata()
            self._verify_cache_integrity()
            
        except Exception as e:
            logger.error(f"Cache initialization failed: {str(e)}")
            raise

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata with validation."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                if self._validate_metadata_structure(metadata):
                    return metadata
            except json.JSONDecodeError:
                logger.warning("Corrupted metadata file, creating new")
                
        return {
            'entries': {},
            'total_size': 0,
            'last_cleanup': time.time(),
            'version': '1.0'
        }

    def _validate_metadata_structure(self, metadata: Dict) -> bool:
        """Validate metadata structure."""
        required_keys = {'entries', 'total_size', 'last_cleanup'}
        return all(key in metadata for key in required_keys)

    def _verify_cache_integrity(self):
        """Verify and repair cache integrity."""
        try:
            actual_files = {
                p.name: p.stat().st_size 
                for p in self.cache_dir.glob("*") 
                if p.name != "cache_metadata.json"
            }
            metadata_files = set(self.metadata['entries'].keys())
            
            # Process in batches for memory efficiency
            batch_size = 1000
            updates = {}
            total_size = 0
            
            # Remove missing files from metadata
            for file_id in metadata_files - set(actual_files.keys()):
                self.metadata['total_size'] -= self.metadata['entries'][file_id]['size']
                del self.metadata['entries'][file_id]
            
            # Add or update existing files
            for file_id, size in actual_files.items():
                if file_id not in metadata_files:
                    updates[file_id] = {
                        'path': str(self.cache_dir / file_id),
                        'size': size,
                        'last_access': time.time(),
                        'access_count': 0,
                        'created': time.time()
                    }
                total_size += size
                
                if len(updates) >= batch_size:
                    self._batch_update_metadata(updates)
                    updates.clear()
            
            if updates:
                self._batch_update_metadata(updates)
            
            self.metadata['total_size'] = total_size
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Cache integrity verification failed: {str(e)}")
            raise

    def _batch_update_metadata(self, updates: Dict[str, Any]):
        """Update metadata in batches."""
        with self.lock:
            self.metadata['entries'].update(updates)

    def _start_monitoring_tasks(self):
        """Start monitoring tasks."""
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._memory_monitor())

    async def _cleanup_loop(self):
        """Periodic cache cleanup."""
        while True:
            try:
                await self._cleanup_cache()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _memory_monitor(self):
        """Monitor system and GPU memory."""
        while True:
            try:
                # Check system memory
                sys_memory = psutil.virtual_memory().percent / 100
                
                # Check GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if gpu_memory > self.memory_threshold:
                        await self._emergency_cleanup('gpu')
                
                if sys_memory > self.memory_threshold:
                    await self._emergency_cleanup('system')
                
                await asyncio.sleep(self.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitor error: {str(e)}")
                await asyncio.sleep(60)

    async def _emergency_cleanup(self, trigger: str):
        """Emergency cleanup when memory threshold exceeded."""
        logger.warning(f"Emergency cleanup triggered by {trigger} memory usage")
        
        try:
            # Aggressive cleanup
            with self.lock:
                entries = list(self.metadata['entries'].items())
                entries.sort(key=lambda x: (x[1]['access_count'], x[1]['last_access']))
                
                # Remove up to 25% of cache entries
                target_count = len(entries) // 4
                for file_id, _ in entries[:target_count]:
                    await self._remove_entry(file_id)
            
            if trigger == 'gpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {str(e)}")

    async def _cleanup_cache(self):
        """Regular cache cleanup."""
        with self.lock:
            try:
                current_time = time.time()
                entries = self.metadata['entries']
                
                # Process expired entries
                expired = [
                    file_id for file_id, info in entries.items()
                    if current_time - info['last_access'] > self.ttl
                ]
                
                for file_id in expired:
                    await self._remove_entry(file_id)
                
                # Process size limit
                if self.metadata['total_size'] > self.max_size:
                    remaining = [
                        (file_id, info) for file_id, info in entries.items()
                        if file_id not in expired
                    ]
                    
                    remaining.sort(
                        key=lambda x: (
                            x[1]['access_count'],
                            x[1]['last_access']
                        )
                    )
                    
                    for file_id, _ in remaining:
                        if self.metadata['total_size'] <= self.max_size:
                            break
                        await self._remove_entry(file_id)
                
                self.metadata['last_cleanup'] = current_time
                await self._async_save_metadata()
                
            except Exception as e:
                logger.error(f"Cache cleanup failed: {str(e)}")

    async def _remove_entry(self, file_id: str):
        """Remove cache entry with async I/O."""
        try:
            info = self.metadata['entries'][file_id]
            file_path = Path(info['path'])
            
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)
            
            self.metadata['total_size'] -= info['size']
            del self.metadata['entries'][file_id]
            
        except Exception as e:
            logger.error(f"Error removing cache entry {file_id}: {str(e)}")

    async def _async_save_metadata(self):
        """Save metadata asynchronously."""
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(self.metadata, indent=2))
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")

    def _sync_save_metadata(self):
        """Save metadata synchronously."""
        with self.lock:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

    def get_path(self, key: str) -> Path:
        """Get cache path for key."""
        file_id = self._hash_key(key)
        return self.cache_dir / file_id

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve item from cache."""
        file_id = self._hash_key(key)
        
        with self.lock:
            if file_id in self.metadata['entries']:
                entry = self.metadata['entries'][file_id]
                entry['last_access'] = time.time()
                entry['access_count'] += 1
                
                file_path = Path(entry['path'])
                if file_path.exists():
                    asyncio.create_task(self._async_save_metadata())
                    return {
                        'path': file_path,
                        'metadata': entry
                    }
        
        return None

    async def put(
        self,
        key: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store item in cache with type-specific handling."""
        file_id = self._hash_key(key)
        file_path = self.cache_dir / file_id
        
        try:
            if isinstance(data, (torch.Tensor, torch.nn.Module)):
                await asyncio.to_thread(torch.save, data, file_path)
            elif isinstance(data, (dict, list)):
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(data))
            else:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(data)
                    
            size = file_path.stat().st_size
            
            with self.lock:
                self.metadata['entries'][file_id] = {
                    'path': str(file_path),
                    'size': size,
                    'last_access': time.time(),
                    'access_count': 0,
                    'metadata': metadata or {},
                    'created': time.time()
                }
                self.metadata['total_size'] += size
                
            await self._async_save_metadata()
            
            if self.metadata['total_size'] > self.max_size:
                asyncio.create_task(self._cleanup_cache())
                
        except Exception as e:
            logger.error(f"Cache storage error for {key}: {str(e)}")
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink)
            raise

    @staticmethod
    def _hash_key(key: str) -> str:
        """Generate deterministic hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def clear(self):
        """Clear entire cache with async I/O."""
        with self.lock:
            try:
                files = list(self.cache_dir.glob("*"))
                for file_path in files:
                    if file_path.name != "cache_metadata.json":
                        await asyncio.to_thread(file_path.unlink)

                self.metadata = {
                    'entries': {},
                    'total_size': 0,
                    'last_cleanup': time.time(),
                    'version': self.metadata.get('version', '1.0')
                }
                await self._async_save_metadata()
                
            except Exception as e:
                logger.error(f"Cache clear error: {str(e)}")
                raise

    async def invalidate(self, key: str):
        """Invalidate specific cache entry."""
        file_id = self._hash_key(key)
        await self._remove_entry(file_id)
        await self._async_save_metadata()

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self.lock:
            stats = {
                'total_size': self.metadata['total_size'],
                'total_entries': len(self.metadata['entries']),
                'max_size': self.max_size,
                'last_cleanup': self.metadata['last_cleanup'],
                'utilization': self.metadata['total_size'] / self.max_size,
                'memory_info': {
                    'system': psutil.virtual_memory().percent,
                    'gpu': None
                }
            }
            
            if torch.cuda.is_available():
                stats['memory_info']['gpu'] = {
                    'allocated': torch.cuda.memory_allocated(),
                    'reserved': torch.cuda.memory_reserved()
                }
                
            return stats

    async def cleanup(self):
        """Perform final cleanup."""
        try:
            await self._cleanup_cache()
            self.thread_pool.shutdown(wait=True)
            logger.info("Cache cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
            raise

    def __del__(self):
        """Cleanup on deletion."""
        self.thread_pool.shutdown(wait=False)