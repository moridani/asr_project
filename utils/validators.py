from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from loguru import logger
import magic
import os
from utils.error_handler import ASRException

class AudioValidator:
    ALLOWED_FORMATS = {
        'wav': ['audio/x-wav', 'audio/wav'],
        'mp3': ['audio/mpeg', 'audio/mp3'],
        'flac': ['audio/flac', 'audio/x-flac'],
        'm4a': ['audio/mp4', 'audio/x-m4a'],
        'ogg': ['audio/ogg', 'application/ogg']
    }

    MIN_DURATION = 0.1  # seconds
    MAX_DURATION = 7200  # 2 hours
    MIN_SAMPLE_RATE = 8000
    MAX_SAMPLE_RATE = 48000
    MIN_BIT_DEPTH = 16

    @classmethod
    def validate_audio_file(
        cls,
        file_path: Path,
        strict: bool = True
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validate audio file with comprehensive checks.
        
        Args:
            file_path: Path to audio file
            strict: If True, raises exceptions for violations
            
        Returns:
            Tuple of (audio_info, warning_message)
        """
        try:
            if not file_path.exists():
                raise ASRException("File not found", "FILE_NOT_FOUND")

            # Check file format
            mime_type = magic.from_file(str(file_path), mime=True)
            format_valid = False
            file_format = None

            for fmt, mime_types in cls.ALLOWED_FORMATS.items():
                if mime_type in mime_types:
                    format_valid = True
                    file_format = fmt
                    break

            if not format_valid:
                raise ASRException(
                    f"Unsupported audio format: {mime_type}",
                    "INVALID_FORMAT"
                )

            # Load and validate audio properties
            audio_info = cls._get_audio_info(file_path, file_format)
            warnings = []

            # Validate duration
            duration = audio_info['duration']
            if duration < cls.MIN_DURATION:
                msg = f"Audio duration ({duration:.2f}s) is too short"
                if strict:
                    raise ASRException(msg, "INVALID_DURATION")
                warnings.append(msg)
            elif duration > cls.MAX_DURATION:
                msg = f"Audio duration ({duration:.2f}s) exceeds maximum"
                if strict:
                    raise ASRException(msg, "INVALID_DURATION")
                warnings.append(msg)

            # Validate sample rate
            sample_rate = audio_info['sample_rate']
            if not cls.MIN_SAMPLE_RATE <= sample_rate <= cls.MAX_SAMPLE_RATE:
                msg = f"Invalid sample rate: {sample_rate}Hz"
                if strict:
                    raise ASRException(msg, "INVALID_SAMPLE_RATE")
                warnings.append(msg)

            # Check audio quality
            quality_issues = cls._check_audio_quality(file_path, audio_info)
            warnings.extend(quality_issues)

            return audio_info, "; ".join(warnings) if warnings else None

        except Exception as e:
            if isinstance(e, ASRException):
                raise
            raise ASRException(f"Audio validation failed: {str(e)}", "VALIDATION_ERROR")

    @classmethod
    def _get_audio_info(cls, file_path: Path, format: str) -> Dict[str, Any]:
        """Get comprehensive audio file information."""
        try:
            if format == 'mp3':
                audio = AudioSegment.from_mp3(str(file_path))
                info = {
                    'format': format,
                    'channels': audio.channels,
                    'sample_rate': audio.frame_rate,
                    'duration': len(audio) / 1000.0,
                    'bit_depth': audio.sample_width * 8,
                    'size_bytes': os.path.getsize(file_path)
                }
            else:
                sf_info = sf.info(str(file_path))
                info = {
                    'format': format,
                    'channels': sf_info.channels,
                    'sample_rate': sf_info.samplerate,
                    'duration': sf_info.duration,
                    'bit_depth': sf_info.subtype_info[1],
                    'size_bytes': os.path.getsize(file_path)
                }

            # Calculate bitrate
            info['bitrate'] = int(info['size_bytes'] * 8 / info['duration'])
            
            return info

        except Exception as e:
            raise ASRException(f"Failed to read audio info: {str(e)}", "READ_ERROR")

    @classmethod
    def _check_audio_quality(
        cls,
        file_path: Path,
        audio_info: Dict[str, Any]
    ) -> list:
        """Check for audio quality issues."""
        warnings = []

        try:
            # Load a small sample for quality checks
            if audio_info['format'] == 'mp3':
                audio = AudioSegment.from_mp3(str(file_path))
                samples = np.array(audio.get_array_of_samples())
            else:
                samples, _ = sf.read(str(file_path), frames=int(audio_info['sample_rate']))

            # Check for clipping
            if np.abs(samples).max() >= 0.99:
                warnings.append("Audio contains clipping")

            # Check for low volume
            rms = np.sqrt(np.mean(samples**2))
            if rms < 0.1:
                warnings.append("Audio volume might be too low")

            # Check DC offset
            dc_offset = np.mean(samples)
            if abs(dc_offset) > 0.1:
                warnings.append("Significant DC offset detected")

            # Check bit depth
            if audio_info['bit_depth'] < cls.MIN_BIT_DEPTH:
                warnings.append(f"Low bit depth: {audio_info['bit_depth']} bits")

            # Check bitrate for compressed formats
            if audio_info['format'] in ['mp3', 'm4a']:
                min_bitrate = 128000  # 128 kbps
                if audio_info['bitrate'] < min_bitrate:
                    warnings.append(f"Low bitrate: {audio_info['bitrate']/1000:.0f} kbps")

        except Exception as e:
            logger.warning(f"Quality check failed: {str(e)}")
            warnings.append("Could not perform full quality analysis")

        return warnings

def validate_audio_file(file_path: str, strict: bool = True) -> Dict[str, Any]:
    """Convenience function for audio validation."""
    audio_info, warnings = AudioValidator.validate_audio_file(
        Path(file_path),
        strict
    )
    
    if warnings:
        logger.warning(f"Audio validation warnings: {warnings}")
        
    return {
        **audio_info,
        'warnings': warnings
    }