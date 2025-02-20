from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator, conint, confloat
import torch
from pathlib import Path

class HardwareConfig(BaseModel):
    device: str
    num_threads: conint(ge=1) = 4
    batch_size: conint(ge=1) = 8

    @validator('device')
    def validate_device(cls, v):
        if v == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available")
        if v not in ['cuda', 'cpu']:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        return v

class ModelConfig(BaseModel):
    whisper_model: str
    french_model: str
    nllb_model: str
    diarization_model: str
    hf_token: Optional[str]

    @validator('whisper_model')
    def validate_whisper_model(cls, v):
        valid_models = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']
        if v not in valid_models:
            raise ValueError(f"Whisper model must be one of {valid_models}")
        return v

class ProcessingConfig(BaseModel):
    sample_rate: conint(ge=8000, le=48000) = 16000
    min_speakers: conint(ge=1) = 1
    max_speakers: conint(ge=1) = 10
    min_segment_length: confloat(ge=0.1) = 0.5
    max_segment_length: confloat(ge=1.0) = 30.0
    overlap_threshold: confloat(ge=0.0, le=1.0) = 0.5

    @validator('max_speakers')
    def validate_speakers(cls, v, values):
        if 'min_speakers' in values and v < values['min_speakers']:
            raise ValueError("max_speakers must be greater than min_speakers")
        return v

    @validator('max_segment_length')
    def validate_segment_length(cls, v, values):
        if 'min_segment_length' in values and v < values['min_segment_length']:
            raise ValueError("max_segment_length must be greater than min_segment_length")
        return v

class CacheConfig(BaseModel):
    max_cache_size: conint(ge=1024*1024*1024) = 10*1024*1024*1024  # min 1GB
    cache_ttl: conint(ge=3600) = 3600*24  # min 1 hour
    clean_cache_interval: conint(ge=300) = 3600  # min 5 minutes

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: conint(ge=1, le=65535) = 8000
    workers: conint(ge=1) = 4
    max_upload_size: conint(ge=1024*1024) = 100*1024*1024  # min 1MB
    allowed_audio_formats: List[str] = ["mp3", "wav", "m4a", "ogg", "flac"]
    enable_auth: bool = False
    api_key: Optional[str] = None

    @validator('api_key')
    def validate_api_key(cls, v, values):
        if values.get('enable_auth', False) and not v:
            raise ValueError("API key must be provided when authentication is enabled")
        return v

class PathConfig(BaseModel):
    project_root: Path
    cache_dir: Path
    model_dir: Path
    upload_dir: Path
    results_dir: Path
    log_dir: Path

    @validator('*')
    def validate_paths(cls, v):
        if isinstance(v, Path):
            if not v.parent.exists():
                raise ValueError(f"Parent directory {v.parent} does not exist")
        return v

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete configuration."""
    try:
        # Validate each configuration section
        hardware_config = HardwareConfig(**config.get('hardware', {}))
        model_config = ModelConfig(**config.get('model', {}))
        processing_config = ProcessingConfig(**config.get('processing', {}))
        cache_config = CacheConfig(**config.get('cache', {}))
        api_config = APIConfig(**config.get('api', {}))
        path_config = PathConfig(**config.get('paths', {}))

        # Combine validated configs
        validated_config = {
            'hardware': hardware_config.dict(),
            'model': model_config.dict(),
            'processing': processing_config.dict(),
            'cache': cache_config.dict(),
            'api': api_config.dict(),
            'paths': path_config.dict()
        }

        # Additional cross-section validations
        _validate_cross_section(validated_config)

        return validated_config

    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")

def _validate_cross_section(config: Dict[str, Any]):
    """Validate relationships between different configuration sections."""
    # Validate CUDA memory requirements
    if config['hardware']['device'] == 'cuda':
        required_memory = _estimate_cuda_memory_requirements(config)
        available_memory = torch.cuda.get_device_properties(0).total_memory
        if required_memory > available_memory:
            raise ValueError(
                f"Estimated CUDA memory requirement ({required_memory/1e9:.2f}GB) "
                f"exceeds available memory ({available_memory/1e9:.2f}GB)"
            )

    # Validate cache size vs available disk space
    cache_path = Path(config['paths']['cache_dir'])
    if cache_path.exists():
        import shutil
        total, _, free = shutil.disk_usage(cache_path)
        if config['cache']['max_cache_size'] > free:
            raise ValueError(
                f"Configured cache size ({config['cache']['max_cache_size']/1e9:.2f}GB) "
                f"exceeds available disk space ({free/1e9:.2f}GB)"
            )

def _estimate_cuda_memory_requirements(config: Dict[str, Any]) -> int:
    """Estimate CUDA memory requirements based on configuration."""
    model_memory_estimates = {
        'tiny': 1e9,      # 1GB
        'base': 1.5e9,    # 1.5GB
        'small': 2e9,     # 2GB
        'medium': 5e9,    # 5GB
        'large-v1': 10e9, # 10GB
        'large-v2': 10e9, # 10GB
        'large-v3': 10e9  # 10GB
    }

    # Base memory for Whisper model
    total_memory = model_memory_estimates.get(
        config['model']['whisper_model'].split('-')[0],
        10e9
    )

    # Additional memory for other models
    total_memory += 2e9  # NLLB model
    total_memory += 1e9  # Diarization model
    total_memory += 1e9  # French model

    # Memory for batch processing
    total_memory *= 1.5  # 50% overhead for batch processing

    return int(total_memory)