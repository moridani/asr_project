from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ConfigDict
import torch
from loguru import logger
import sys
import platform
import psutil

class ModelConfig(BaseModel):
    """Model-specific configuration."""
    model_config = ConfigDict(extra='forbid')

    name: str
    path: str
    language: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class HardwareConfig(BaseModel):
    """Hardware-specific configuration."""
    model_config = ConfigDict(extra='forbid')

    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    num_threads: int = Field(default=4, ge=1)
    batch_size: int = Field(default=8, ge=1)
    gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)

class ASRSettings(BaseModel):
    """ASR System Configuration."""
    model_config = ConfigDict(
        extra='forbid',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True
    )

    # Project paths
    PROJECT_ROOT: Path = Field(default=Path(__file__).parent.parent)
    CACHE_DIR: Path = Field(default=None)
    MODEL_DIR: Path = Field(default=None)
    UPLOAD_DIR: Path = Field(default=None)
    RESULTS_DIR: Path = Field(default=None)
    LOG_DIR: Path = Field(default=None)

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1, le=65535)
    API_WORKERS: int = Field(default=4, ge=1)
    API_TIMEOUT: int = Field(default=300, ge=30)  # 5 minutes minimum
    MAX_UPLOAD_SIZE: int = Field(default=1024 * 1024 * 100)  # 100MB
    ALLOWED_AUDIO_FORMATS: List[str] = Field(
        default=["mp3", "wav", "m4a", "ogg", "flac"]
    )
    ALLOWED_MIME_TYPES: List[str] = Field(
        default=[
            "audio/mpeg", "audio/wav", "audio/x-wav",
            "audio/mp4", "audio/ogg", "audio/flac"
        ]
    )

    # Hardware Configuration
    HARDWARE: HardwareConfig = Field(default_factory=HardwareConfig)

    # Model Configurations
    WHISPER_MODEL: str = Field(default="large-v3")
    FRENCH_MODEL: str = Field(default="Helsinki-NLP/opus-mt-fr-en")
    NLLB_MODEL: str = Field(default="facebook/nllb-200-1.3B")
    DIARIZATION_MODEL: str = Field(default="pyannote/speaker-diarization-3.1")
    
    # Processing Parameters
    SAMPLE_RATE: int = Field(default=16000, ge=8000, le=48000)
    MIN_SPEAKERS: int = Field(default=1, ge=1)
    MAX_SPEAKERS: int = Field(default=10, ge=1)
    MIN_SEGMENT_LENGTH: float = Field(default=0.5, ge=0.1)
    MAX_SEGMENT_LENGTH: float = Field(default=30.0, ge=1.0)
    OVERLAP_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)

    # Performance Settings
    MAX_CONCURRENT_TASKS: int = Field(default=5, ge=1)
    TASK_TIMEOUT: int = Field(default=3600, ge=300)  # 1 hour default
    CHUNK_SIZE: int = Field(default=1024 * 1024, ge=1024)  # 1MB chunks

    # Cache Configuration
    MAX_CACHE_SIZE: int = Field(default=10 * 1024 * 1024 * 1024)  # 10GB
    CACHE_TTL: int = Field(default=3600 * 24)  # 24 hours
    CLEAN_CACHE_INTERVAL: int = Field(default=3600)  # 1 hour

    # Authentication
    HF_TOKEN: Optional[str] = None
    ENABLE_AUTH: bool = Field(default=False)
    API_KEY: Optional[str] = None
    JWT_SECRET_KEY: Optional[str] = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1)

    # Redis Configuration (optional)
    USE_REDIS: bool = Field(default=False)
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535)
    REDIS_DB: int = Field(default=0, ge=0)

    @field_validator('PROJECT_ROOT')
    @classmethod
    def validate_project_root(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Project root directory does not exist: {v}")
        return v.absolute()

    @field_validator('CACHE_DIR', 'MODEL_DIR', 'UPLOAD_DIR', 'RESULTS_DIR', 'LOG_DIR')
    @classmethod
    def validate_directories(cls, v: Optional[Path], info: Dict[str, Any]) -> Path:
        if v is None:
            # Get PROJECT_ROOT from values if available
            project_root = info.data.get('PROJECT_ROOT', Path(__file__).parent.parent)
            # Use directory name from field name
            dir_name = info.field_name.lower().replace('_dir', '')
            return project_root / dir_name
        return v.absolute()

    @field_validator('MAX_SPEAKERS')
    @classmethod
    def validate_max_speakers(cls, v: int, info: Dict[str, Any]) -> int:
        min_speakers = info.data.get('MIN_SPEAKERS', 1)
        if v < min_speakers:
            raise ValueError(f"MAX_SPEAKERS ({v}) must be >= MIN_SPEAKERS ({min_speakers})")
        return v

    @field_validator('MAX_SEGMENT_LENGTH')
    @classmethod
    def validate_max_segment_length(cls, v: float, info: Dict[str, Any]) -> float:
        min_length = info.data.get('MIN_SEGMENT_LENGTH', 0.5)
        if v < min_length:
            raise ValueError(
                f"MAX_SEGMENT_LENGTH ({v}) must be >= MIN_SEGMENT_LENGTH ({min_length})"
            )
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_directories()
        self._init_logging()
        self._validate_system_requirements()

    def _init_directories(self) -> None:
        """Initialize required directories."""
        try:
            for directory in [
                self.CACHE_DIR,
                self.MODEL_DIR,
                self.UPLOAD_DIR,
                self.RESULTS_DIR,
                self.LOG_DIR
            ]:
                directory.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            raise RuntimeError(f"Failed to create directories: {str(e)}")

    def _init_logging(self) -> None:
        """Initialize logging configuration."""
        try:
            logger.remove()  # Remove default handler
            logger.add(
                self.LOG_DIR / "asr.log",
                rotation="100 MB",
                retention="30 days",
                level="INFO",
                backtrace=True,
                diagnose=True
            )
            logger.add(sys.stderr, level="WARNING")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize logging: {str(e)}")

    def _validate_system_requirements(self) -> None:
        """Validate system requirements."""
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")

        # Check available memory
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
            raise RuntimeError("Insufficient memory available (min 4GB required)")

        # Check disk space
        disk = psutil.disk_usage(self.PROJECT_ROOT)
        if disk.free < self.MAX_CACHE_SIZE * 2:
            raise RuntimeError("Insufficient disk space available")

        # Check CUDA availability if requested
        if self.HARDWARE.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.HARDWARE.device = "cpu"

        # Validate authentication settings
        if self.ENABLE_AUTH:
            if not self.API_KEY and not self.JWT_SECRET_KEY:
                raise RuntimeError("Either API_KEY or JWT_SECRET_KEY must be set when auth is enabled")

        # Check HuggingFace token
        if not self.HF_TOKEN:
            logger.warning("HF_TOKEN not set. Some models may not be available.")

    def get_model_path(self, model_name: str) -> Path:
        """Get path for model cache."""
        safe_name = "".join(c if c.isalnum() else '_' for c in model_name)
        return self.MODEL_DIR / safe_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.model_dump().items()
        }

    def save_config(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file."""
        try:
            if path is None:
                path = self.PROJECT_ROOT / "config.json"
                
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {str(e)}")

    @classmethod
    def load_config(cls, path: Path) -> 'ASRSettings':
        """Load configuration from file."""
        try:
            with open(path) as f:
                config = json.load(f)
            return cls(**config)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }

# Global settings instance
settings = ASRSettings()