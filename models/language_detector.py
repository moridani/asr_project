from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import importlib.util
import time

# Check required packages
required_packages = {
    'speechbrain': 'speechbrain',
    'faster_whisper': 'faster-whisper',
}

missing_packages = []
for package, pip_name in required_packages.items():
    if importlib.util.find_spec(package) is None:
        missing_packages.append(f"{pip_name}")

if missing_packages:
    raise ImportError(
        f"Required packages are missing. Please install: pip install {' '.join(missing_packages)}"
    )

# Import after validation
from speechbrain.pretrained import EncoderClassifier
from faster_whisper import WhisperModel

class DetectionException(Exception):
    """Custom exception for language detection errors."""
    pass

class LanguageDetector:
    """Language detection with optimized performance for multilingual audio."""
    
    SUPPORTED_LANGUAGES = {
        'fr': 'French',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'en': 'English'
    }
    
    def __init__(
        self,
        device: torch.device,
        cache_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize language detection.
        
        Args:
            device: Computing device (CPU/GPU)
            cache_manager: Model cache manager
            config: Optional configuration dictionary
        """
        try:
            self.device = device
            self.cache_manager = cache_manager
            self.config = self._validate_config(config or {})
            
            # Initialize models
            self._initialize_models()
            
            # Thread pool for CPU tasks
            self.thread_pool = ThreadPoolExecutor(max_workers=2)
            
            logger.info(f"Language detector initialized on {device}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise DetectionException(f"Initialization failed: {str(e)}")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default configuration."""
        defaults = {
            'whisper_model_size': 'medium',
            'min_confidence': 0.2,
            'window_size': 3.0,
            'overlap': 1.0,
            'batch_size': 8,
            'primary_model_weight': 0.7
        }
        return {**defaults, **config}

    def _initialize_models(self) -> None:
        """Initialize detection models."""
        try:
            # Primary model for accurate language detection
            self.voxlingua = self._init_voxlingua()
            
            # Secondary model for verification
            self.whisper = self._init_whisper()
            
            logger.info("Language detection models initialized")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise DetectionException(f"Model initialization failed: {str(e)}")

    def _init_voxlingua(self) -> EncoderClassifier:
        """Initialize VoxLingua model."""
        try:
            model_path = self.cache_manager.get_path("voxlingua107")
            return EncoderClassifier.from_hparams(
                source="TalTechNLP/voxlingua107-epaca-tdnn",
                savedir=str(model_path),
                run_opts={"device": self.device}
            )
        except Exception as e:
            raise DetectionException(f"VoxLingua initialization failed: {str(e)}")

    def _init_whisper(self) -> WhisperModel:
        """Initialize Whisper model."""
        try:
            compute_type = (
                "float16" 
                if self.device.type == "cuda" 
                else "float32"
            )
            
            return WhisperModel(
                self.config['whisper_model_size'],
                device=self.device.type,
                compute_type=compute_type,
                num_workers=1
            )
        except Exception as e:
            raise DetectionException(f"Whisper initialization failed: {str(e)}")

    async def detect(
        self,
        audio: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Detect languages in audio.
        
        Args:
            audio: Input audio tensor/array
            
        Returns:
            Dict containing language detection results
        """
        try:
            start_time = time.time()
            
            # Convert numpy array to tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            
            # Ensure audio is on correct device
            audio = audio.to(self.device)
            
            # Run detectors in parallel
            primary_scores, secondary_scores = await asyncio.gather(
                self._detect_with_voxlingua(audio),
                self._detect_with_whisper(audio)
            )
            
            # Combine results
            combined_results = self._combine_results(
                primary_scores,
                secondary_scores
            )
            
            # Generate segments
            segments = await self._generate_segments(audio, combined_results)
            
            processing_time = time.time() - start_time
            
            return {
                'primary_language': combined_results[0]['language'],
                'distribution': combined_results[:3],  # Top 3 languages
                'confidence': combined_results[0]['confidence'],
                'segments': segments,
                'processing_time': processing_time,
                'models_used': ['voxlingua', 'whisper']
            }
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            raise DetectionException(f"Detection failed: {str(e)}")

    async def _detect_with_voxlingua(
        self,
        audio: torch.Tensor
    ) -> Dict[str, float]:
        """Primary language detection."""
        try:
            def _process():
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    predictions = self.voxlingua.classify_batch(audio)
                    scores = torch.nn.functional.softmax(predictions[0], dim=0)
                    return {
                        lang: score.item()
                        for lang, score in zip(
                            self.voxlingua.hparams.label_encoder.classes_,
                            scores
                        )
                    }
            
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                _process
            )
        except Exception as e:
            logger.error(f"VoxLingua detection failed: {str(e)}")
            return {}

    async def _detect_with_whisper(
        self,
        audio: torch.Tensor
    ) -> Dict[str, float]:
        """Secondary language detection."""
        try:
            def _process():
                audio_np = audio.cpu().numpy()
                segments, _ = self.whisper.transcribe(
                    audio_np,
                    task="lang_id"
                )
                return {
                    segment.language: np.exp(segment.avg_logprob)
                    for segment in segments
                }
            
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                _process
            )
        except Exception as e:
            logger.error(f"Whisper detection failed: {str(e)}")
            return {}

    def _combine_results(
        self,
        primary_scores: Dict[str, float],
        secondary_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Combine and validate detection results."""
        try:
            combined = {}
            primary_weight = self.config['primary_model_weight']
            secondary_weight = 1 - primary_weight
            
            # Process all detected languages
            all_languages = set(primary_scores) | set(secondary_scores)
            
            for lang in all_languages:
                # Get scores with fallback to 0
                primary_score = primary_scores.get(lang, 0)
                secondary_score = secondary_scores.get(lang, 0)
                
                # Calculate weighted score
                score = (primary_score * primary_weight) + (secondary_score * secondary_weight)
                
                # Calculate agreement
                agreement = sum([
                    1 for scores in [primary_scores, secondary_scores]
                    if lang in scores and scores[lang] > self.config['min_confidence']
                ]) / 2
                
                if score > self.config['min_confidence']:
                    combined[lang] = {
                        'language': lang,
                        'language_name': self.SUPPORTED_LANGUAGES.get(lang, 'Unknown'),
                        'confidence': score,
                        'agreement': agreement,
                        'model_scores': {
                            'primary': primary_score,
                            'secondary': secondary_score
                        }
                    }
            
            # Sort by confidence and agreement
            return sorted(
                combined.values(),
                key=lambda x: (x['confidence'] * (1 + x['agreement'])),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Result combination failed: {str(e)}")
            return []

    async def _generate_segments(
        self,
        audio: torch.Tensor,
        language_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate time-stamped language segments."""
        if not language_results:
            return []
            
        try:
            segments = []
            window_size = self.config['window_size']
            overlap = self.config['overlap']
            sample_rate = self.voxlingua.hparams.sample_rate
            
            # Process audio in windows
            audio_length = len(audio) / sample_rate
            num_windows = int((audio_length - overlap) / (window_size - overlap))
            
            async def process_window(start_idx: int) -> Optional[Dict[str, Any]]:
                start_time = start_idx * (window_size - overlap)
                end_time = min(start_time + window_size, audio_length)
                
                # Extract window
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                window = audio[start_sample:end_sample]
                
                # Detect language in window
                scores = await self._detect_with_voxlingua(window)
                
                if scores:
                    max_lang = max(scores.items(), key=lambda x: x[1])
                    return {
                        'start': start_time,
                        'end': end_time,
                        'language': max_lang[0],
                        'language_name': self.SUPPORTED_LANGUAGES.get(max_lang[0], 'Unknown'),
                        'confidence': max_lang[1]
                    }
                return None
            
            # Process windows in parallel
            window_results = await asyncio.gather(*[
                process_window(i) for i in range(num_windows)
            ])
            
            # Filter and merge segments
            valid_segments = [s for s in window_results if s is not None]
            return self._merge_segments(valid_segments)
            
        except Exception as e:
            logger.error(f"Segment generation failed: {str(e)}")
            return []

    def _merge_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge adjacent segments with same language."""
        if not segments:
            return []
            
        try:
            merged = []
            current = segments[0].copy()
            
            for next_seg in segments[1:]:
                if (next_seg['language'] == current['language'] and
                    next_seg['start'] - current['end'] < 0.5):
                    # Merge segments
                    current['end'] = next_seg['end']
                    current['confidence'] = (
                        current['confidence'] + next_seg['confidence']
                    ) / 2
                else:
                    merged.append(current)
                    current = next_seg.copy()
            
            merged.append(current)
            return merged
            
        except Exception as e:
            logger.error(f"Segment merging failed: {str(e)}")
            return segments

    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Language detector cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")