from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import numpy as np
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import importlib.util
import time
import gc

# Check required packages
required_packages = {
    'speechbrain': 'speechbrain',
    'faster_whisper': 'faster-whisper',
    'transformers': 'transformers',
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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn.functional as F

class LanguageDetectionResult:
    """Structured class for language detection results"""
    def __init__(
        self,
        primary_language: str,
        distribution: List[Dict[str, Any]],
        confidence: float,
        segments: List[Dict[str, Any]],
        processing_time: float,
        models_used: List[str]
    ):
        self.primary_language = primary_language
        self.distribution = distribution
        self.confidence = confidence
        self.segments = segments
        self.processing_time = processing_time
        self.models_used = models_used
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'primary_language': self.primary_language,
            'distribution': self.distribution,
            'confidence': self.confidence,
            'segments': self.segments,
            'processing_time': self.processing_time,
            'models_used': self.models_used
        }

class DetectionException(Exception):
    """Custom exception for language detection errors."""
    pass

class LanguageDetector:
    """Enhanced language detection with optimized performance for multilingual audio."""
    
    SUPPORTED_LANGUAGES = {
        'fr': 'French',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'en': 'English',
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    def __init__(
        self,
        device: torch.device,
        cache_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize language detection with optimized settings.
        
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
            
            # Thread pool for CPU tasks (limit workers to avoid memory issues)
            self.thread_pool = ThreadPoolExecutor(max_workers=2)
            
            # Progress tracking
            self.progress_callback = None
            self.detection_progress = 0.0
            
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
            'primary_model_weight': 0.7,
            'min_duration': 1.0,
            'support_threshold': 0.3,
            'max_memory_usage': 0.8,  # Maximum memory fraction to use
            'optimize_for_french': True,  # Special optimization for French
            'segment_batch_size': 10,
        }
        
        # Update defaults with provided config
        validated_config = {**defaults, **config}
        
        # Validate values
        if validated_config['window_size'] < 0.5:
            validated_config['window_size'] = 0.5
        if validated_config['batch_size'] < 1:
            validated_config['batch_size'] = 1
        if not 0 <= validated_config['primary_model_weight'] <= 1:
            validated_config['primary_model_weight'] = 0.7
            
        return validated_config

    def _initialize_models(self) -> None:
        """Initialize detection models with proper memory management."""
        try:
            # Primary model for accurate language detection
            self.voxlingua = self._init_voxlingua()
            
            # Secondary model for verification
            self.whisper = self._init_whisper()
            
            # French-specific model for enhanced French detection
            self.french_model = None
            if self.config['optimize_for_french']:
                self.french_model = self._init_french_model()
            
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
        """Initialize Whisper model with memory optimization."""
        try:
            compute_type = (
                "float16" 
                if self.device.type == "cuda" 
                else "int8"  # Use quantized model for CPU
            )
            
            # Use disk cache to reduce memory usage
            return WhisperModel(
                self.config['whisper_model_size'],
                device=self.device.type,
                compute_type=compute_type,
                download_root=str(self.cache_manager.get_path("whisper")),
                cpu_threads=4,  # Limit CPU threads
                num_workers=1    # Limit workers
            )
        except Exception as e:
            raise DetectionException(f"Whisper initialization failed: {str(e)}")
    
    def _init_french_model(self) -> Dict[str, Any]:
        """Initialize French-specific model."""
        try:
            model_id = "facebook/wav2vec2-large-xlsr-53-french"
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            
            # Load to CPU first, then transfer to device to better manage memory
            model = Wav2Vec2ForCTC.from_pretrained(model_id)
            model = model.to(self.device)
            
            return {"model": model, "processor": processor}
            
        except Exception as e:
            logger.warning(f"French model initialization failed: {str(e)}")
            return None

    async def detect(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        progress_callback: Optional[callable] = None,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Detect languages in audio with progress tracking.
        
        Args:
            audio: Input audio tensor/array
            progress_callback: Optional callback for progress updates
            sample_rate: Audio sample rate
            
        Returns:
            Dict containing language detection results
        """
        try:
            start_time = time.time()
            self.progress_callback = progress_callback
            self.detection_progress = 0.0
            
            # Update progress
            if self.progress_callback:
                await self.progress_callback(0.0, "Starting language detection")
            
            # Convert numpy array to tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio.astype(np.float32))
            
            # Check if audio meets minimum requirements
            if len(audio) < self.config['min_duration'] * sample_rate:
                raise DetectionException("Audio too short for reliable language detection")
            
            # Ensure audio is on correct device
            audio = audio.to(self.device)
            
            # Memory management - clear unused variables
            await self._manage_memory()
            
            # Update progress
            await self._update_progress(0.1, "Running language detectors")
            
            # Run detectors in parallel
            primary_scores, secondary_scores, french_scores = await asyncio.gather(
                self._detect_with_voxlingua(audio),
                self._detect_with_whisper(audio),
                self._detect_with_french_model(audio) if self.french_model else asyncio.sleep(0, result={})
            )
            
            # Update progress
            await self._update_progress(0.5, "Analyzing results")
            
            # Combine results
            combined_results = self._combine_results(
                primary_scores,
                secondary_scores,
                french_scores
            )
            
            # Update progress
            await self._update_progress(0.8, "Generating segments")
            
            # Generate segments
            segments = await self._generate_segments(audio, combined_results, sample_rate)
            
            # Final result
            processing_time = time.time() - start_time
            result = LanguageDetectionResult(
                primary_language=combined_results[0]['language'],
                distribution=combined_results[:5],  # Top 5 languages
                confidence=combined_results[0]['confidence'],
                segments=segments,
                processing_time=processing_time,
                models_used=['voxlingua', 'whisper'] + (['french_wav2vec'] if self.french_model else [])
            )
            
            # Update progress
            await self._update_progress(1.0, "Language detection complete")
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            raise DetectionException(f"Detection failed: {str(e)}")
        finally:
            # Clear memory
            await self._manage_memory()

    async def _update_progress(self, progress: float, status: str) -> None:
        """Update detection progress."""
        self.detection_progress = progress
        if self.progress_callback:
            await self.progress_callback(progress, status)

    async def _manage_memory(self) -> None:
        """Manage memory usage to prevent OOM errors."""
        try:
            # Clear CUDA cache if using GPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Memory management failed: {str(e)}")

    async def _detect_with_voxlingua(
        self,
        audio: torch.Tensor
    ) -> Dict[str, float]:
        """Primary language detection using VoxLingua."""
        try:
            def _process():
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    # Process in smaller chunks for long audio
                    if len(audio) > 30 * 16000:  # 30 seconds
                        chunk_size = 20 * 16000  # 20 seconds chunks
                        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
                        chunk_predictions = []
                        
                        for i in range(num_chunks):
                            start_idx = i * chunk_size
                            end_idx = min(start_idx + chunk_size, len(audio))
                            audio_chunk = audio[start_idx:end_idx].unsqueeze(0)
                            predictions = self.voxlingua.classify_batch(audio_chunk)
                            chunk_predictions.append(predictions)
                            
                        # Average predictions
                        avg_prediction = torch.mean(torch.stack(chunk_predictions), dim=0)
                        scores = torch.nn.functional.softmax(avg_prediction[0], dim=0)
                    else:
                        # Process normally for shorter audio
                        predictions = self.voxlingua.classify_batch(audio.unsqueeze(0))
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
        """Secondary language detection using Whisper."""
        try:
            def _process():
                audio_np = audio.cpu().numpy()
                
                try:
                    # Using language identification with faster-whisper
                    segments, info = self.whisper.transcribe(
                        audio_np,
                        task="lang_id",
                        beam_size=1  # Use smaller beam size for faster processing
                    )
                    
                    # Build result
                    result = {}
                    segment_list = list(segments)  # Convert iterator to list
                    
                    if segment_list:
                        # Get detected languages with probabilities
                        for segment in segment_list:
                            result[segment.language] = np.exp(segment.avg_logprob)
                            
                    return result
                except Exception as e:
                    logger.warning(f"Whisper detection internal error: {str(e)}")
                    return {}
            
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                _process
            )
        except Exception as e:
            logger.error(f"Whisper detection failed: {str(e)}")
            return {}

    async def _detect_with_french_model(
        self,
        audio: torch.Tensor
    ) -> Dict[str, float]:
        """French-specific detection."""
        if not self.french_model:
            return {}
            
        try:
            def _process():
                # Process using wav2vec2 French model
                audio_np = audio.cpu().numpy()
                inputs = self.french_model["processor"](
                    audio_np, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.french_model["model"](**inputs)
                    logits = outputs.logits
                    
                # Calculate confidence based on CTC output probabilities
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1)[0].mean().item()
                
                # Only return if confidence is high enough
                if confidence > 0.6:
                    return {'fr': confidence}
                else:
                    return {}
                    
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                _process
            )
        except Exception as e:
            logger.warning(f"French model detection failed: {str(e)}")
            return {}

    def _combine_results(
        self,
        primary_scores: Dict[str, float],
        secondary_scores: Dict[str, float],
        french_scores: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """Combine and validate detection results with French optimization."""
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
                
                # Calculate agreement level between models
                agreement = sum([
                    1 for scores in [primary_scores, secondary_scores]
                    if lang in scores and scores[lang] > self.config['min_confidence']
                ]) / 2  # Normalize to 0-1 range
                
                # Only include if score is above threshold
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
            
            # Apply French-specific optimization if available
            if french_scores and 'fr' in french_scores:
                french_confidence = french_scores['fr']
                
                # Boost French score if detected by specialist model
                if 'fr' in combined:
                    # Weighted combination favoring specialist model
                    combined['fr']['confidence'] = (
                        combined['fr']['confidence'] * 0.4 + 
                        french_confidence * 0.6
                    )
                    combined['fr']['agreement'] = max(combined['fr']['agreement'], 0.8)
                    combined['fr']['model_scores']['french_wav2vec'] = french_confidence
                elif french_confidence > 0.7:
                    # Add French if not detected by other models but high confidence
                    combined['fr'] = {
                        'language': 'fr',
                        'language_name': 'French',
                        'confidence': french_confidence * 0.8,  # Slightly discount
                        'agreement': 0.5,  # Medium agreement since only one model
                        'model_scores': {
                            'primary': 0,
                            'secondary': 0,
                            'french_wav2vec': french_confidence
                        }
                    }
            
            # Sort by confidence and agreement
            return sorted(
                combined.values(),
                key=lambda x: (x['confidence'] * (1 + x['agreement'] * 0.5)),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Result combination failed: {str(e)}")
            return []

    async def _generate_segments(
        self,
        audio: torch.Tensor,
        language_results: List[Dict[str, Any]],
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """Generate time-stamped language segments with batched processing."""
        if not language_results:
            return []
            
        try:
            # Process audio in windows
            window_size = int(self.config['window_size'] * sample_rate)
            overlap = int(self.config['overlap'] * sample_rate)
            step_size = window_size - overlap
            
            # Calculate number of windows
            audio_length = len(audio)
            num_windows = (audio_length - overlap) // step_size if step_size > 0 else 1
            
            # Process in batches for memory efficiency
            batch_size = self.config['segment_batch_size']
            segments = []
            
            for batch_start in range(0, num_windows, batch_size):
                batch_end = min(batch_start + batch_size, num_windows)
                batch_tasks = []
                
                for i in range(batch_start, batch_end):
                    start_sample = i * step_size
                    end_sample = min(start_sample + window_size, audio_length)
                    
                    # Skip if window is too small
                    if end_sample - start_sample < sample_rate * 0.5:
                        continue
                        
                    window = audio[start_sample:end_sample]
                    batch_tasks.append(self._process_window(
                        window, 
                        start_sample / sample_rate,
                        end_sample / sample_rate
                    ))
                
                # Process batch concurrently
                batch_results = await asyncio.gather(*batch_tasks)
                segments.extend([s for s in batch_results if s])
            
            # Merge adjacent segments with same language
            merged_segments = self._merge_segments(segments)
            
            # Refine segment boundaries using signal energy
            refined_segments = await self._refine_segment_boundaries(audio, merged_segments, sample_rate)
            
            return refined_segments
            
        except Exception as e:
            logger.error(f"Segment generation failed: {str(e)}")
            return []

    async def _process_window(
        self,
        window: torch.Tensor,
        start_time: float,
        end_time: float
    ) -> Optional[Dict[str, Any]]:
        """Process a single window for language detection."""
        try:
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
        except Exception as e:
            logger.warning(f"Window processing failed: {str(e)}")
            return None

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
                    # Update confidence (weighted by duration)
                    current_duration = current['end'] - current['start']
                    next_duration = next_seg['end'] - next_seg['start']
                    total_duration = current_duration + next_duration
                    
                    current['confidence'] = (
                        (current['confidence'] * current_duration) +
                        (next_seg['confidence'] * next_duration)
                    ) / total_duration
                else:
                    # Start new segment if different language or gap
                    if current['end'] - current['start'] >= self.config['min_duration']:
                        merged.append(current)
                    current = next_seg.copy()
            
            # Add last segment if long enough
            if current['end'] - current['start'] >= self.config['min_duration']:
                merged.append(current)
                
            return merged
            
        except Exception as e:
            logger.error(f"Segment merging failed: {str(e)}")
            return segments

    async def _refine_segment_boundaries(
        self,
        audio: torch.Tensor,
        segments: List[Dict[str, Any]],
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Refine segment boundaries using signal energy."""
        if not segments:
            return []
            
        try:
            refined_segments = []
            
            for segment in segments:
                # Convert time to samples
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Ensure within audio bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                
                if end_sample <= start_sample:
                    continue
                
                # Extract segment audio
                segment_audio = audio[start_sample:end_sample].cpu().numpy()
                
                # Compute energy
                energy = np.abs(segment_audio) ** 2
                
                # Find true start (first point above threshold)
                energy_threshold = 0.05 * np.max(energy)
                true_start = start_sample
                for i in range(len(energy)):
                    if energy[i] > energy_threshold:
                        true_start = start_sample + i
                        break
                
                # Find true end (last point above threshold)
                true_end = end_sample
                for i in range(len(energy) - 1, -1, -1):
                    if energy[i] > energy_threshold:
                        true_end = start_sample + i + 1
                        break
                
                # Convert back to time
                refined_start = true_start / sample_rate
                refined_end = true_end / sample_rate
                
                # Only keep if length is sufficient
                if refined_end - refined_start >= self.config['min_duration']:
                    refined_segment = segment.copy()
                    refined_segment['start'] = refined_start
                    refined_segment['end'] = refined_end
                    refined_segments.append(refined_segment)
            
            return refined_segments
            
        except Exception as e:
            logger.warning(f"Boundary refinement failed: {str(e)}")
            return segments

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Free up memory
            self.french_model = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            gc.collect()
            
            logger.info("Language detector cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")