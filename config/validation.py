from typing import Dict, Any, List, Optional, Union, Set, Tuple
from pydantic import BaseModel, field_validator, ConfigDict, Field
import torch
from pathlib import Path
import json
import os
import shutil
import psutil
import numpy as np
from loguru import logger

class HardwareConfig(BaseModel):
    """Hardware configuration with enhanced validation."""
    model_config = ConfigDict(extra='forbid')
    
    device: str
    num_threads: int = Field(default=4, ge=1)
    batch_size: int = Field(default=8, ge=1)
    gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)
    cpu_priority: str = Field(default="normal")

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device with detailed error messages."""
        if v == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA device specified but not available - falling back to CPU")
                return 'cpu'
            return v
        elif v == 'cpu':
            return v
        elif v.startswith('cuda:'):
            if not torch.cuda.is_available():
                logger.warning("CUDA device specified but not available - falling back to CPU")
                return 'cpu'
            
            try:
                device_idx = int(v.split(':')[1])
                if device_idx >= torch.cuda.device_count():
                    logger.warning(f"CUDA device {device_idx} not found (max={torch.cuda.device_count()-1}) - using device 0")
                    return 'cuda:0'
                return v
            except ValueError:
                logger.warning(f"Invalid CUDA device format: {v} - using default device")
                return 'cuda'
        else:
            raise ValueError(f"Device must be 'cpu', 'cuda', or 'cuda:N', got '{v}'")
    
    @field_validator('cpu_priority')
    @classmethod
    def validate_cpu_priority(cls, v: str) -> str:
        """Validate CPU priority setting."""
        valid_priorities = ['low', 'normal', 'high']
        if v.lower() not in valid_priorities:
            raise ValueError(f"CPU priority must be one of {valid_priorities}, got '{v}'")
        return v.lower()

class ModelConfig(BaseModel):
    """Model configuration with enhanced validation."""
    model_config = ConfigDict(extra='forbid')
    
    whisper_model: str
    french_model: str
    nllb_model: str
    diarization_model: str
    hf_token: Optional[str] = None
    load_strategy: str = Field(default="lazy")
    quantization: Optional[str] = None
    optimize_for_french: bool = Field(default=True)
    
    @field_validator('whisper_model')
    @classmethod
    def validate_whisper_model(cls, v: str) -> str:
        """Validate Whisper model with enhanced size checking."""
        valid_models = [
            'tiny', 'base', 'small', 'medium', 
            'large-v1', 'large-v2', 'large-v3',
            'tiny.en', 'base.en', 'small.en', 'medium.en'
        ]
        
        # Direct match
        if v in valid_models:
            return v
            
        # Check for custom model path
        if os.path.exists(v):
            return v
            
        # Check for HuggingFace model ID format
        if '/' in v and not os.path.exists(v):
            logger.warning(f"Model '{v}' will be loaded from HuggingFace")
            return v
            
        raise ValueError(f"Whisper model must be one of {valid_models} or a valid model path/ID, got '{v}'")

    @field_validator('load_strategy')
    @classmethod
    def validate_load_strategy(cls, v: str) -> str:
        """Validate model loading strategy."""
        valid_strategies = ['eager', 'lazy', 'on_demand']
        if v not in valid_strategies:
            raise ValueError(f"Load strategy must be one of {valid_strategies}, got '{v}'")
        return v
    
    @field_validator('quantization')
    @classmethod
    def validate_quantization(cls, v: Optional[str]) -> Optional[str]:
        """Validate quantization setting."""
        if v is None:
            return v
            
        valid_quantizations = ['int8', 'int4', 'fp16', 'bf16']
        if v not in valid_quantizations:
            raise ValueError(f"Quantization must be one of {valid_quantizations}, got '{v}'")
        return v

class ProcessingConfig(BaseModel):
    """Processing configuration with enhanced validation."""
    model_config = ConfigDict(extra='forbid')
    
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    min_speakers: int = Field(default=1, ge=1)
    max_speakers: int = Field(default=10, ge=1)
    min_segment_length: float = Field(default=0.5, ge=0.1)
    max_segment_length: float = Field(default=30.0, ge=1.0)
    overlap_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_threshold: float = Field(default=20.0, ge=5.0, le=60.0)
    language_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    transcription_beam_size: int = Field(default=5, ge=1)
    max_batch_size: int = Field(default=8, ge=1)
    progress_tracking: bool = Field(default=True)
    
    @field_validator('max_speakers')
    @classmethod
    def validate_speakers(cls, v: int, info: Dict[str, Any]) -> int:
        """Validate speaker count relationship."""
        if 'min_speakers' in info.data and v < info.data['min_speakers']:
            raise ValueError(f"max_speakers ({v}) must be greater than or equal to min_speakers ({info.data['min_speakers']})")
        return v

    @field_validator('max_segment_length')
    @classmethod
    def validate_segment_length(cls, v: float, info: Dict[str, Any]) -> float:
        """Validate segment length relationship."""
        if 'min_segment_length' in info.data and v < info.data['min_segment_length']:
            raise ValueError(f"max_segment_length ({v}) must be greater than min_segment_length ({info.data['min_segment_length']})")
        return v

class CacheConfig(BaseModel):
    """Cache configuration with enhanced validation and cleanup strategy."""
    model_config = ConfigDict(extra='forbid')
    
    max_cache_size: int = Field(default=10*1024*1024*1024, ge=1024*1024*1024)  # min 1GB
    cache_ttl: int = Field(default=3600*24, ge=3600)  # min 1 hour
    clean_cache_interval: int = Field(default=3600, ge=300)  # min 5 minutes
    cache_strategy: str = Field(default="lru")
    memory_limit_warning: float = Field(default=0.85, ge=0.5, le=0.95)
    memory_limit_critical: float = Field(default=0.95, ge=0.6, le=0.99)
    
    @field_validator('cache_strategy')
    @classmethod
    def validate_cache_strategy(cls, v: str) -> str:
        """Validate cache strategy."""
        valid_strategies = ['lru', 'lfu', 'fifo']
        if v.lower() not in valid_strategies:
            raise ValueError(f"Cache strategy must be one of {valid_strategies}, got '{v}'")
        return v.lower()

class APIConfig(BaseModel):
    """API configuration with enhanced validation."""
    model_config = ConfigDict(extra='forbid')
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    timeout: int = Field(default=300, ge=30, le=3600)
    max_upload_size: int = Field(default=100*1024*1024, ge=1024*1024)  # min 1MB
    allowed_audio_formats: List[str] = Field(default=["mp3", "wav", "m4a", "ogg", "flac"])
    enable_auth: bool = Field(default=False)
    api_key: Optional[str] = None
    max_concurrent_tasks: int = Field(default=5, ge=1, le=50)
    task_timeout: int = Field(default=3600, ge=300)  # 5 minutes minimum
    cors_settings: Dict[str, Any] = Field(default_factory=dict)
    enable_documentation: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: Dict[str, Any]) -> Optional[str]:
        """Validate API key when auth is enabled."""
        if info.data.get('enable_auth', False) and not v:
            raise ValueError("API key must be provided when authentication is enabled")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}, got '{v}'")
        return v.upper()

class PathConfig(BaseModel):
    """Path configuration with enhanced validation."""
    model_config = ConfigDict(extra='forbid')
    
    project_root: Path
    cache_dir: Path
    model_dir: Path
    upload_dir: Path
    results_dir: Path
    log_dir: Path
    temp_dir: Optional[Path] = None
    custom_vocab_path: Optional[Path] = None

    @field_validator('project_root')
    @classmethod
    def validate_project_root(cls, v: Path) -> Path:
        """Validate project root directory."""
        v = Path(v).resolve()
        if not v.exists():
            raise ValueError(f"Project root directory does not exist: {v}")
        return v
    
    @field_validator('cache_dir', 'model_dir', 'upload_dir', 'results_dir', 'log_dir')
    @classmethod
    def validate_required_dirs(cls, v: Path, info: Dict[str, Any]) -> Path:
        """Validate required directories or create them."""
        if isinstance(v, str):
            v = Path(v)
            
        # If relative path, make it relative to project_root
        if not v.is_absolute() and 'project_root' in info.data:
            v = info.data['project_root'] / v
            
        # Resolve to absolute path
        v = v.resolve()
        
        # Create directory if it doesn't exist
        if not v.exists():
            try:
                v.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {v}")
            except Exception as e:
                raise ValueError(f"Failed to create directory {v}: {str(e)}")
                
        return v
    
    @field_validator('temp_dir', 'custom_vocab_path')
    @classmethod
    def validate_optional_paths(cls, v: Optional[Path], info: Dict[str, Any]) -> Optional[Path]:
        """Validate optional paths."""
        if v is None:
            return None
            
        if isinstance(v, str):
            v = Path(v)
            
        # If relative path, make it relative to project_root
        if not v.is_absolute() and 'project_root' in info.data:
            v = info.data['project_root'] / v
            
        # For custom_vocab_path, check that it exists
        if info.field_name == 'custom_vocab_path' and not v.exists():
            raise ValueError(f"Custom vocabulary file does not exist: {v}")
            
        # For temp_dir, create if it doesn't exist
        if info.field_name == 'temp_dir' and not v.exists():
            try:
                v.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Failed to create temp directory {v}: {str(e)}")
                
        return v.resolve()

class ASRConfig(BaseModel):
    """Complete ASR system configuration."""
    model_config = ConfigDict(extra='forbid')
    
    hardware: HardwareConfig
    model: ModelConfig
    processing: ProcessingConfig
    cache: CacheConfig
    api: APIConfig
    paths: PathConfig
    version: str = Field(default="1.0.0")
    description: str = Field(default="ASR System Configuration")
    maintainer: Optional[str] = None
    french_optimization: bool = Field(default=True)

def validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate configuration from file."""
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ValueError(f"Configuration file does not exist: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Validate configuration using pydantic model
        validated_config = ASRConfig(**config_data)
        
        # Additional cross-validation
        cross_validate_config(validated_config)
        
        # Return validated config as dict
        return validated_config.model_dump()
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise

def cross_validate_config(config: ASRConfig) -> None:
    """Perform cross-section validation checks."""
    # Validate CUDA memory requirements if using GPU
    if config.hardware.device.startswith('cuda'):
        _validate_gpu_requirements(config)
        
    # Validate disk space requirements
    _validate_disk_requirements(config)
    
    # Validate model compatibility
    _validate_model_compatibility(config)

def _validate_gpu_requirements(config: ASRConfig) -> None:
    """Validate GPU memory requirements."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA device specified but not available")
        
    # Get available GPU memory
    gpu_id = 0
    if config.hardware.device.startswith('cuda:'):
        try:
            gpu_id = int(config.hardware.device.split(':')[1])
        except (IndexError, ValueError):
            gpu_id = 0
            
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} not available (max index: {torch.cuda.device_count()-1})")
        
    available_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    
    # Estimate memory requirements based on models
    required_memory = _estimate_model_memory(config.model)
    
    # Check if available memory is sufficient
    if required_memory > available_memory * config.hardware.gpu_memory_fraction:
        raise ValueError(
            f"Estimated memory requirement ({required_memory/1e9:.2f}GB) "
            f"exceeds available GPU memory limit "
            f"({available_memory * config.hardware.gpu_memory_fraction/1e9:.2f}GB)"
        )

def _validate_disk_requirements(config: ASRConfig) -> None:
    """Validate disk space requirements."""
    # Check cache directory
    cache_path = config.paths.cache_dir
    if cache_path.exists():
        try:
            total, used, free = shutil.disk_usage(cache_path)
            
            # Check if cache size exceeds available space
            if config.cache.max_cache_size > free:
                logger.warning(
                    f"Configured cache size ({config.cache.max_cache_size/1e9:.2f}GB) "
                    f"exceeds available disk space ({free/1e9:.2f}GB)"
                )
                
        except Exception as e:
            logger.warning(f"Failed to check disk usage: {str(e)}")
    
    # Check upload directory
    upload_path = config.paths.upload_dir
    if upload_path.exists():
        try:
            total, used, free = shutil.disk_usage(upload_path)
            
            # Check if upload limit is reasonable
            if config.api.max_upload_size * config.api.max_concurrent_tasks > free * 0.5:
                logger.warning(
                    f"Maximum concurrent uploads "
                    f"({config.api.max_upload_size * config.api.max_concurrent_tasks/1e9:.2f}GB) "
                    f"may exceed available disk space ({free/1e9:.2f}GB)"
                )
                
        except Exception as e:
            logger.warning(f"Failed to check disk usage: {str(e)}")

def _validate_model_compatibility(config: ASRConfig) -> None:
    """Validate model compatibility."""
    # Check if French optimization requires French model
    if config.french_optimization and not any(
        model_name in config.model.french_model.lower()
        for model_name in ['french', 'fr']
    ):
        logger.warning(
            f"French optimization is enabled but French model '{config.model.french_model}' "
            "may not be optimized for French"
        )
    
    # Check if Whisper model size is appropriate for the hardware
    if config.hardware.device == 'cpu' and config.model.whisper_model.startswith('large'):
        logger.warning(
            f"Using large Whisper model ({config.model.whisper_model}) on CPU may be slow. "
            "Consider using a smaller model or GPU acceleration."
        )

def _estimate_model_memory(model_config: ModelConfig) -> int:
    """Estimate memory requirements for models in bytes."""
    # Memory estimates for Whisper models (bytes)
    whisper_sizes = {
        'tiny': 0.5e9,      # 0.5GB
        'base': 1.0e9,      # 1GB
        'small': 2.0e9,     # 2GB
        'medium': 5.0e9,    # 5GB
        'large-v1': 10.0e9, # 10GB
        'large-v2': 10.0e9, # 10GB
        'large-v3': 10.0e9  # 10GB
    }
    
    # Handle whisper model sizes
    if model_config.whisper_model in whisper_sizes:
        whisper_memory = whisper_sizes[model_config.whisper_model]
    elif model_config.whisper_model.split('-')[0] in whisper_sizes:
        whisper_memory = whisper_sizes[model_config.whisper_model.split('-')[0]]
    elif model_config.whisper_model.split('.')[0] in whisper_sizes:
        whisper_memory = whisper_sizes[model_config.whisper_model.split('.')[0]]
    else:
        # Default to large estimate for unknown models
        whisper_memory = 10.0e9
    
    # Additional memory for other models
    french_model_memory = 2.0e9   # 2GB estimate for French model
    diarization_memory = 1.5e9    # 1.5GB for diarization
    nllb_model_memory = 2.5e9     # 2.5GB for NLLB translation
    
    # Apply memory optimization if quantization is enabled
    if model_config.quantization:
        if model_config.quantization == 'int8':
            whisper_memory *= 0.5  # ~50% reduction
            french_model_memory *= 0.5
            nllb_model_memory *= 0.5
        elif model_config.quantization == 'int4':
            whisper_memory *= 0.25  # ~75% reduction
            french_model_memory *= 0.25
            nllb_model_memory *= 0.25
    
    # Total estimated memory
    total_memory = whisper_memory + french_model_memory + diarization_memory + nllb_model_memory
    
    # Add overhead for processing
    total_memory *= 1.2  # 20% overhead
    
    return int(total_memory)

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    # Determine project root
    project_root = Path.cwd()
    
    # Create default configuration
    default_config = {
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_threads": min(os.cpu_count() or 4, 8),
            "batch_size": 8,
            "gpu_memory_fraction": 0.9,
            "cpu_priority": "normal"
        },
        "model": {
            "whisper_model": "large-v3",
            "french_model": "facebook/wav2vec2-large-xlsr-53-french",
            "nllb_model": "facebook/nllb-200-1.3B",
            "diarization_model": "pyannote/speaker-diarization-3.1",
            "hf_token": None,
            "load_strategy": "lazy",
            "quantization": None,
            "optimize_for_french": True
        },
        "processing": {
            "sample_rate": 16000,
            "min_speakers": 1,
            "max_speakers": 10,
            "min_segment_length": 0.5,
            "max_segment_length": 30.0,
            "overlap_threshold": 0.5,
            "silence_threshold": 20.0,
            "language_detection_confidence": 0.5,
            "transcription_beam_size": 5,
            "max_batch_size": 8,
            "progress_tracking": True
        },
        "cache": {
            "max_cache_size": 10 * 1024 * 1024 * 1024,  # 10GB
            "cache_ttl": 3600 * 24,  # 24 hours
            "clean_cache_interval": 3600,  # 1 hour
            "cache_strategy": "lru",
            "memory_limit_warning": 0.85,
            "memory_limit_critical": 0.95
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "timeout": 300,
            "max_upload_size": 100 * 1024 * 1024,  # 100MB
            "allowed_audio_formats": ["mp3", "wav", "m4a", "ogg", "flac"],
            "enable_auth": False,
            "api_key": None,
            "max_concurrent_tasks": 5,
            "task_timeout": 3600,
            "cors_settings": {
                "allow_origins": ["*"],
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            },
            "enable_documentation": True,
            "log_level": "INFO"
        },
        "paths": {
            "project_root": str(project_root),
            "cache_dir": str(project_root / "cache"),
            "model_dir": str(project_root / "models"),
            "upload_dir": str(project_root / "uploads"),
            "results_dir": str(project_root / "results"),
            "log_dir": str(project_root / "logs"),
            "temp_dir": str(project_root / "temp"),
            "custom_vocab_path": None
        },
        "version": "1.0.0",
        "description": "Default ASR System Configuration",
        "maintainer": None,
        "french_optimization": True
    }
    
    return default_config

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    try:
        config_path = Path(config_path)
        
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}")
        raise

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate configuration from file."""
    return validate_config(config_path)