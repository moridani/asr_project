from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from pydantic import BaseModel, Field, validator
import aiofiles
import aiohttp
import asyncio
import uuid
import psutil
import importlib
from pathlib import Path
import json
import time
import shutil
import logging
from loguru import logger
import sys

# Optional GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .auth import (
    auth_manager,
    get_current_user,
    requires_permissions,
    AuthData,
    Token,
    AuthException
)
from core.pipeline import ASRPipeline
from config.settings import settings
from utils.validators import validate_audio_file
from utils.error_handler import handle_error, ASRException

# Define all models
class ProcessingRequest(BaseModel):
    file_id: str
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    webhook_url: Optional[str] = None
    priority: Optional[int] = Field(default=0, ge=0, le=10)

    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        if v is not None and not v.startswith(('http://', 'https://')):
            raise ValueError('Webhook URL must start with http:// or https://')
        return v

class ProcessingResponse(BaseModel):
    task_id: str
    status: str
    file_id: str
    webhook_url: Optional[str] = None
    estimated_time: Optional[float] = None

class TaskLog(BaseModel):
    timestamp: datetime
    level: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

# Global state with type hints
active_tasks: Dict[str, Dict[str, Any]] = {}
pipeline: Optional[ASRPipeline] = None

async def cleanup_temp_files() -> None:
    """Clean up temporary files and directories."""
    try:
        # Clean up upload directory
        upload_dir = Path(settings.UPLOAD_DIR)
        if upload_dir.exists():
            for file_path in upload_dir.glob("*"):
                if file_path.is_file():
                    await aiofiles.os.remove(file_path)
                
        # Clean up results directory
        results_dir = Path(settings.RESULTS_DIR)
        if results_dir.exists():
            for file_path in results_dir.glob("*"):
                if file_path.is_file():
                    await aiofiles.os.remove(file_path)
                
        logger.info("Temporary files cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")
        raise

async def send_webhook_update(
    webhook_url: str,
    task_id: str,
    data: Optional[Dict[str, Any]] = None
) -> None:
    """Send webhook updates for task status."""
    if task_id not in active_tasks:
        logger.error(f"Task {task_id} not found for webhook update")
        return

    try:
        task = active_tasks[task_id]
        payload = {
            'task_id': task_id,
            'status': task['status'],
            'progress': task['progress'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if data:
            payload['data'] = data
            
        timeout = aiohttp.ClientTimeout(total=10)  # 10 seconds timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status >= 400:
                    logger.error(
                        f"Webhook failed for {task_id}: "
                        f"Status {response.status}, "
                        f"Response: {await response.text()}"
                    )
                    
    except asyncio.TimeoutError:
        logger.error(f"Webhook timeout for {task_id}")
    except Exception as e:
        logger.error(f"Webhook error for {task_id}: {str(e)}")

        
async def process_audio_task(
    task_id: str,
    file_path: Path,
    config: Dict[str, Any],
    progress_callback: Optional[callable] = None,
    webhook_url: Optional[str] = None,
    user_id: Optional[str] = None
) -> None:
    """Process audio file in background with progress tracking."""
    if not pipeline:
        raise ASRException("Pipeline not initialized")

    try:
        active_tasks[task_id]['status'] = 'processing'
        
        # Use the new progress callback parameter
        result = await pipeline.process_file(
            str(file_path),
            config=config,
            progress_callback=progress_callback
        )     # Add metadata
        result['metadata'].update({
            'task_id': task_id,
            'user_id': user_id,
            'processing_time': time.time() - active_tasks[task_id]['start_time'],
            'config': config,
            'file_path': str(file_path),
            'completion_time': datetime.now(timezone.utc).isoformat()
        })
        
        # Save results
        result_path = Path(settings.RESULTS_DIR) / f"{task_id}.json"
        async with aiofiles.open(result_path, 'w') as f:
            await f.write(json.dumps(result, indent=2))
            
        active_tasks[task_id].update({
            'status': 'completed',
            'progress': 1.0,
            'completion_time': time.time()
        })
        
        if webhook_url:
            await send_webhook_update(webhook_url, task_id, result)
            
    except Exception as e:
        error_info = handle_error(e, {'task_id': task_id})
        active_tasks[task_id].update({
            'status': 'failed',
            'error': error_info,
            'completion_time': time.time()
        })
        
        if webhook_url:
            await send_webhook_update(webhook_url, task_id, error_info)
        
        logger.error(f"Task {task_id} failed: {str(e)}")
        
        # Cleanup on failure
        try:
            if file_path.exists():
                await aiofiles.os.remove(file_path)
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed for {task_id}: {str(cleanup_error)}")

async def delete_task_implementation(
    task_id: str,
    auth: AuthData
) -> Dict[str, str]:
    """Implementation of task deletion."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = active_tasks[task_id]
    
    # Check access permission
    if task['user_id'] != auth.user_id and 'admin' not in auth.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this task"
        )
        
    try:
        # Remove result file if exists
        result_path = Path(settings.RESULTS_DIR) / f"{task_id}.json"
        if result_path.exists():
            await aiofiles.os.remove(result_path)
            
        # Remove uploaded file if exists
        file_id = task['file_id']
        upload_pattern = Path(settings.UPLOAD_DIR) / f"{file_id}.*"
        for file_path in Path(settings.UPLOAD_DIR).glob(f"{file_id}.*"):
            await aiofiles.os.remove(file_path)
            
        # Remove task from active tasks
        del active_tasks[task_id]
        
        return {"status": "deleted"}
        
    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    # Startup
    global pipeline
    try:
        pipeline = ASRPipeline(
            device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
            config=settings.model_config
        )
        logger.info(f"ASR Pipeline initialized on {pipeline.device}")
        
        # Create necessary directories
        for directory in [settings.UPLOAD_DIR, settings.RESULTS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    try:
        if pipeline:
            await pipeline.cleanup()
            
        # Cleanup temporary files
        for task_id, task in active_tasks.items():
            if task['status'] == 'processing':
                logger.warning(f"Task {task_id} was interrupted during shutdown")
                
        await cleanup_temp_files()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=None if settings.DISABLE_DOCS else "/docs",
    redoc_url=None if settings.DISABLE_DOCS else "/redoc",
    lifespan=lifespan
)
# [Previous imports and initial setup remain the same...]

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
    expose_headers=["X-Rate-Limit-Remaining", "X-Rate-Limit-Reset", "X-Request-ID"]
)

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """Obtain JWT token for authentication."""
    try:
        # In production, validate against database
        if form_data.username == settings.TEST_USER and \
           form_data.password == settings.TEST_PASSWORD:
            
            access_token = await auth_manager.create_access_token(
                data={"sub": form_data.username}
            )
            refresh_token = await auth_manager.create_refresh_token(
                form_data.username
            )
            
            return Token(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
            
        raise AuthException(
            status_code=401,
            detail="Incorrect username or password",
            error_code="INVALID_CREDENTIALS"
        )
        
    except Exception as e:
        logger.error(f"Token generation failed: {str(e)}")
        raise

@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    auth: AuthData = Depends(get_current_user)
) -> Dict[str, str]:
    """Upload audio file for processing."""
    try:
        # Check file size early
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE/1024/1024:.1f}MB"
            )
        
        # Validate file type
        content_type = file.content_type.lower()
        if not any(content_type.startswith(t) for t in ['audio/', 'video/']):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {content_type}"
            )
        
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file extension: {file_extension}"
            )
            
        file_path = Path(settings.UPLOAD_DIR) / f"{file_id}{file_extension}"
        
        # Save file with progress tracking
        file_size = 0
        chunk_size = settings.UPLOAD_CHUNK_SIZE
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(chunk_size):
                await out_file.write(chunk)
                file_size += len(chunk)
                
                if file_size > settings.MAX_UPLOAD_SIZE:
                    await aiofiles.os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE/1024/1024:.1f}MB"
                    )
        
        # Validate audio file
        try:
            validation_result = await validate_audio_file(str(file_path))
            if validation_result.get('warnings'):
                logger.warning(
                    f"Audio validation warnings for {file_id}: "
                    f"{validation_result['warnings']}"
                )
        except Exception as e:
            await aiofiles.os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(e))
        
        # Log upload
        logger.info(
            f"File uploaded by user {auth.user_id}: {file_id} "
            f"(size: {file_size/1024/1024:.2f}MB, type: {content_type})"
        )
            
        return {
            "file_id": file_id,
            "size": file_size,
            "content_type": content_type,
            "filename": file.filename,
            "warnings": validation_result.get('warnings')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessingResponse)
@requires_permissions(["write"])
async def process_audio(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    auth: AuthData = Depends(get_current_user)
) -> ProcessingResponse:
    """Initialize audio processing task."""
    try:
        # Validate file existence
        files = list(Path(settings.UPLOAD_DIR).glob(f"{request.file_id}.*"))
        if not files:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        file_path = files[0]
        
        # Check concurrent tasks limit
        user_tasks = sum(
            1 for task in active_tasks.values()
            if task.get('user_id') == auth.user_id and
            task['status'] in ['queued', 'processing']
        )
        if user_tasks >= settings.MAX_CONCURRENT_TASKS:
            raise HTTPException(
                status_code=429,
                detail=f"Too many concurrent tasks. Maximum: {settings.MAX_CONCURRENT_TASKS}"
            )
            
        task_id = str(uuid.uuid4())
        
        # Estimate processing time based on file size and complexity
        file_size = file_path.stat().st_size
        estimated_time = estimate_processing_time(
            file_size,
            request.config.get('quality_level', 'medium')
        )
        
        # Validate and prepare configuration
        config = validate_and_prepare_config(request.config)
        
        active_tasks[task_id] = {
            'status': 'queued',
            'progress': 0.0,
            'start_time': time.time(),
            'file_id': request.file_id,
            'config': config,
            'webhook_url': request.webhook_url,
            'user_id': auth.user_id,
            'priority': request.priority,
            'estimated_time': estimated_time,
            'file_path': str(file_path),
            'file_size': file_size
        }
        
        background_tasks.add_task(
            process_audio_task,
            task_id,
            file_path,
            config,
            request.webhook_url,
            auth.user_id
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status='queued',
            file_id=request.file_id,
            webhook_url=request.webhook_url,
            estimated_time=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
@requires_permissions(["read"])
async def get_status(
    task_id: str,
    auth: AuthData = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get task processing status."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = active_tasks[task_id]
    
    # Check access permission
    if task['user_id'] != auth.user_id and 'admin' not in auth.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this task"
        )
        
    response = {
        'status': task['status'],
        'progress': task['progress'],
        'details': task.get('status_detail'),
        'estimated_time': task.get('estimated_time'),
        'elapsed_time': time.time() - task['start_time'],
        'remaining_time': None
    }
    
    # Calculate remaining time
    if task['status'] == 'processing' and task['progress'] > 0:
        elapsed_time = time.time() - task['start_time']
        remaining_time = (elapsed_time / task['progress']) * (1 - task['progress'])
        response['remaining_time'] = remaining_time
    
    if task['status'] == 'failed':
        response['error'] = task.get('error')
        
    # Add resource usage if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        response['resources'] = {
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
            'gpu_memory_cached': torch.cuda.memory_reserved() / 1024 / 1024  # MB
        }
        
    return response

@app.get("/result/{task_id}")
@requires_permissions(["read"])
async def get_result(
    task_id: str,
    auth: AuthData = Depends(get_current_user)
) -> JSONResponse:
    """Get processing results."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = active_tasks[task_id]
    
    # Check access permission
    if task['user_id'] != auth.user_id and 'admin' not in auth.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this task"
        )
        
    if task['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Task is {task['status']}"
        )
        
    result_path = Path(settings.RESULTS_DIR) / f"{task_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
        
    try:
        async with aiofiles.open(result_path, 'r') as f:
            content = await f.read()
            return JSONResponse(content=json.loads(content))
    except Exception as e:
        logger.error(f"Failed to read result file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read result file")

@app.delete("/task/{task_id}")
@requires_permissions(["write"])
async def delete_task(
    task_id: str,
    auth: AuthData = Depends(get_current_user)
) -> Dict[str, str]:
    """Delete task and associated files."""
    return await delete_task_implementation(task_id, auth)

# Health Check Endpoints
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check system health and resource usage."""
    try:
        # Collect system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': {
                'pipeline': pipeline is not None,
                'gpu': TORCH_AVAILABLE and torch.cuda.is_available(),
                'disk_space': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            },
            'resources': {
                'active_tasks': len(active_tasks),
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                },
                'process': {
                    'memory_info': dict(psutil.Process().memory_info()._asdict()),
                    'cpu_percent': psutil.Process().cpu_percent()
                }
            },
            'tasks': {
                'queued': sum(1 for t in active_tasks.values() if t['status'] == 'queued'),
                'processing': sum(1 for t in active_tasks.values() if t['status'] == 'processing'),
                'completed': sum(1 for t in active_tasks.values() if t['status'] == 'completed'),
                'failed': sum(1 for t in active_tasks.values() if t['status'] == 'failed')
            }
        }
        
        # Add GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            health_info['resources']['gpu'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'max_memory_allocated': torch.cuda.max_memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            }
        
        # Determine overall status
        if memory.percent > 90 or disk.percent > 90 or cpu_percent > 90:
            health_info['status'] = 'degraded'
        
        return health_info
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )

@app.get("/health/detailed")
@requires_permissions(["admin"])
async def detailed_health_check(
    auth: AuthData = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed system health information."""
    try:
        basic_health = await health_check()
        
        # Add detailed metrics
        detailed_info = {
            **basic_health,
            'system': {
                'boot_time': datetime.fromtimestamp(psutil.boot_time(), timezone.utc).isoformat(),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': dict(psutil.cpu_freq()._asdict()) if hasattr(psutil, 'cpu_freq') else None,
                'load_avg': psutil.getloadavg(),
                'network': {
                    interface: dict(stats._asdict())
                    for interface, stats in psutil.net_if_stats().items()
                }
            },
            'process': {
                'create_time': datetime.fromtimestamp(
                    psutil.Process().create_time(),
                    timezone.utc
                ).isoformat(),
                'cpu_times': dict(psutil.Process().cpu_times()._asdict()),
                'num_threads': psutil.Process().num_threads(),
                'open_files': len(psutil.Process().open_files()),
                'connections': len(psutil.Process().connections())
            },
            'tasks_detailed': {
                task_id: {
                    'status': task['status'],
                    'progress': task['progress'],
                    'start_time': datetime.fromtimestamp(
                        task['start_time'],
                        timezone.utc
                    ).isoformat(),
                    'elapsed_time': time.time() - task['start_time'],
                    'file_size': task.get('file_size'),
                    'user_id': task['user_id']
                }
                for task_id, task in active_tasks.items()
            }
        }
        
        return detailed_info
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics Endpoints
@app.get("/stats")
@requires_permissions(["admin"])
async def get_statistics(
    auth: AuthData = Depends(get_current_user),
    time_range: Optional[str] = "24h"
) -> Dict[str, Any]:
    """Get system statistics and metrics."""
    try:
        # Calculate time range
        end_time = time.time()
        start_time = end_time - {
            "1h": 3600,
            "24h": 86400,
            "7d": 604800,
            "30d": 2592000
        }.get(time_range, 86400)
        
        # Filter tasks within time range
        period_tasks = {
            task_id: task
            for task_id, task in active_tasks.items()
            if task['start_time'] >= start_time
        }
        
        # Calculate statistics
        completed_tasks = [
            task for task in period_tasks.values()
            if task['status'] == 'completed'
        ]
        
        failed_tasks = [
            task for task in period_tasks.values()
            if task['status'] == 'failed'
        ]
        
        stats = {
            'time_range': time_range,
            'start_time': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
            'end_time': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
            'total_tasks': len(period_tasks),
            'tasks_by_status': {
                status: sum(1 for task in period_tasks.values() 
                          if task['status'] == status)
                for status in ['queued', 'processing', 'completed', 'failed']
            },
            'performance': {
                'average_processing_time': (
                    sum(
                        task.get('completion_time', end_time) - task['start_time']
                        for task in completed_tasks
                    ) / len(completed_tasks)
                ) if completed_tasks else 0,
                'error_rate': len(failed_tasks) / len(period_tasks) if period_tasks else 0,
                'success_rate': len(completed_tasks) / len(period_tasks) if period_tasks else 0
            },
            'resource_usage': {
                'total_processed_size': sum(
                    task.get('file_size', 0)
                    for task in completed_tasks
                ),
                'average_file_size': (
                    sum(task.get('file_size', 0) for task in period_tasks.values()) /
                    len(period_tasks)
                ) if period_tasks else 0
            },
            'users': {
                user_id: {
                    'total_tasks': sum(1 for task in period_tasks.values() 
                                     if task['user_id'] == user_id),
                    'completed_tasks': sum(1 for task in completed_tasks 
                                         if task['user_id'] == user_id),
                    'failed_tasks': sum(1 for task in failed_tasks 
                                      if task['user_id'] == user_id),
                    'total_size': sum(task.get('file_size', 0) 
                                    for task in period_tasks.values() 
                                    if task['user_id'] == user_id)
                }
                for user_id in set(task['user_id'] for task in period_tasks.values())
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Statistics generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch Processing Endpoints
@app.post("/tasks/batch")
@requires_permissions(["write"])
async def batch_process(
    request: List[ProcessingRequest],
    background_tasks: BackgroundTasks,
    auth: AuthData = Depends(get_current_user)
) -> List[ProcessingResponse]:
    """Process multiple files in batch."""
    try:
        # Check total batch size
        if len(request) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum ({settings.MAX_BATCH_SIZE})"
            )
        
        # Check user's current tasks
        current_tasks = sum(
            1 for task in active_tasks.values()
            if task['user_id'] == auth.user_id and
            task['status'] in ['queued', 'processing']
        )
        
        if current_tasks + len(request) > settings.MAX_CONCURRENT_TASKS:
            raise HTTPException(
                status_code=429,
                detail=f"Total tasks would exceed maximum ({settings.MAX_CONCURRENT_TASKS})"
            )
        
        responses = []
        for req in request:
            try:
                response = await process_audio(req, background_tasks, auth)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch item failed: {str(e)}")
                responses.append(ProcessingResponse(
                    task_id='',
                    status='failed',
                    file_id=req.file_id,
                    error=str(e)
                ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tasks/batch")
@requires_permissions(["write"])
async def batch_delete(
    task_ids: List[str],
    auth: AuthData = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete multiple tasks."""
    try:
        results = {
            'successful': [],
            'failed': []
        }
        
        for task_id in task_ids:
            try:
                await delete_task_implementation(task_id, auth)
                results['successful'].append(task_id)
            except Exception as e:
                logger.error(f"Failed to delete task {task_id}: {str(e)}")
                results['failed'].append({
                    'task_id': task_id,
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API Documentation
if not settings.DISABLE_DOCS:
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{settings.API_TITLE} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{settings.API_TITLE} - ReDoc",
            redoc_js_url="/static/redoc.standalone.js",
        )

def estimate_processing_time(file_size: int, quality_level: str) -> float:
    """Estimate processing time based on file size and quality settings."""
    # Base processing rate (bytes per second)
    base_rate = {
        'low': 1_000_000,    # 1 MB/s
        'medium': 500_000,   # 500 KB/s
        'high': 250_000      # 250 KB/s
    }.get(quality_level, 500_000)
    
    # Adjust for GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        base_rate *= 3  # GPU is roughly 3x faster
    
    # Add overhead factor
    overhead = 1.2
    
    return (file_size / base_rate) * overhead

def validate_and_prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and prepare processing configuration."""
    default_config = {
        'quality_level': 'medium',
        'noise_reduction': True,
        'diarization': True,
        'language_detection': True,
        'translation': True,
        'max_speakers': 10,
        'min_speaker_time': 1.0
    }
    
    if not config:
        return default_config
        
    # Validate and merge with defaults
    validated_config = default_config.copy()
    validated_config.update({
        k: v for k, v in config.items()
        if k in default_config
    })
    
    # Validate specific fields
    if validated_config['quality_level'] not in ['low', 'medium', 'high']:
        validated_config['quality_level'] = 'medium'
        
    validated_config['max_speakers'] = min(
        max(1, validated_config['max_speakers']),
        20
    )
    
    validated_config['min_speaker_time'] = min(
        max(0.5, validated_config['min_speaker_time']),
        5.0
    )
    
    return validated_config

# Error Handlers
@app.exception_handler(AuthException)
async def auth_exception_handler(
    request: Request,
    exc: AuthException
) -> JSONResponse:
    """Handle authentication exceptions."""
    logger.warning(
        f"Auth error: {exc.detail} "
        f"(code: {exc.error_code}, IP: {request.client.host})"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'error_code': exc.error_code,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.error(
        f"HTTP error: {exc.detail} "
        f"(IP: {request.client.host}, Path: {request.url.path})"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(
        f"Unhandled error: {str(exc)} "
        f"(IP: {request.client.host}, Path: {request.url.path})",
        exc_info=exc
    )
    return JSONResponse(
        status_code=500,
        content={
            'error': "Internal server error",
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    )

# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)

