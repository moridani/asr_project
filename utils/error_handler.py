from typing import Dict, Any, Optional
from loguru import logger
import traceback
import sys
from pathlib import Path
import json
import torch

class ASRException(Exception):
    """Custom exception for ASR pipeline errors."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ErrorHandler:
    ERROR_CODES = {
        'FILE_NOT_FOUND': 'E001',
        'INVALID_FORMAT': 'E002',
        'GPU_ERROR': 'E003',
        'MODEL_LOAD_ERROR': 'E004',
        'PROCESSING_ERROR': 'E005',
        'MEMORY_ERROR': 'E006',
        'CUDA_ERROR': 'E007',
        'VALIDATION_ERROR': 'E008'
    }

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with rotation and formatting."""
        logger.add(
            self.log_dir / "error.log",
            rotation="100 MB",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle and format error response."""
        error_info = self._process_error(error, context)
        self._log_error(error_info)
        return self._format_error_response(error_info)

    def _process_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process and categorize error."""
        error_type = type(error).__name__
        error_code = self._get_error_code(error)
        
        error_info = {
            'error_code': error_code,
            'error_type': error_type,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'severity': self._get_error_severity(error_code)
        }

        if isinstance(error, torch.cuda.CUDAError):
            error_info.update(self._handle_cuda_error(error))
            
        return error_info

    def _get_error_code(self, error: Exception) -> str:
        """Map exception to error code."""
        if isinstance(error, FileNotFoundError):
            return self.ERROR_CODES['FILE_NOT_FOUND']
        elif isinstance(error, ValueError):
            return self.ERROR_CODES['VALIDATION_ERROR']
        elif isinstance(error, torch.cuda.CUDAError):
            return self.ERROR_CODES['CUDA_ERROR']
        elif isinstance(error, torch.cuda.OutOfMemoryError):
            return self.ERROR_CODES['MEMORY_ERROR']
        elif isinstance(error, ASRException):
            return error.error_code or self.ERROR_CODES['PROCESSING_ERROR']
        return self.ERROR_CODES['PROCESSING_ERROR']

    def _get_error_severity(self, error_code: str) -> str:
        """Determine error severity."""
        critical_codes = {'E003', 'E004', 'E006', 'E007'}
        return 'CRITICAL' if error_code in critical_codes else 'ERROR'

    def _handle_cuda_error(self, error: torch.cuda.CUDAError) -> Dict[str, Any]:
        """Handle CUDA-specific errors."""
        return {
            'gpu_info': {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved()
            }
        }

    def _log_error(self, error_info: Dict[str, Any]):
        """Log error with context."""
        log_entry = (
            f"Error Code: {error_info['error_code']} | "
            f"Type: {error_info['error_type']} | "
            f"Message: {error_info['message']}"
        )
        
        if error_info['severity'] == 'CRITICAL':
            logger.critical(log_entry)
        else:
            logger.error(log_entry)
            
        self._save_error_details(error_info)

    def _save_error_details(self, error_info: Dict[str, Any]):
        """Save detailed error information to file."""
        try:
            error_file = self.log_dir / f"error_details_{error_info['error_code']}.json"
            with error_file.open('w') as f:
                json.dump(error_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error details: {str(e)}")

    def _format_error_response(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format error response for API."""
        return {
            'success': False,
            'error': {
                'code': error_info['error_code'],
                'message': error_info['message'],
                'type': error_info['error_type'],
                'severity': error_info['severity']
            }
        }

def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Global error handler function."""
    handler = ErrorHandler()
    return handler.handle_error(error, context)