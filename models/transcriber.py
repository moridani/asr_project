from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import torch
import numpy as np
from faster_whisper import WhisperModel
from transformers import Pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import gc
from tqdm import tqdm
import os
import psutil
import torch.nn.functional as F

class TranscriptionResult:
    """Structured class for transcription results"""
    def __init__(
        self,
        segments: List[Dict[str, Any]],
        text: str,
        confidence: float,
        language: str,
        audio_quality: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        self.segments = segments
        self.text = text
        self.confidence = confidence
        self.language = language
        self.audio_quality = audio_quality
        self.metadata = metadata
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'segments': self.segments,
            'text': self.text,
            'confidence': self.confidence,
            'language': self.language,
            'audio_quality': self.audio_quality,
            'metadata': self.metadata
        }

class TranscriptionException(Exception):
    """Custom exception for transcription errors"""
    pass

class Transcriber:
    """Enhanced transcription with optimized French language support"""
    
    def __init__(
        self,
        device: torch.device,
        cache_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize transcription models with optimizations for French.
        
        Args:
            device: torch device to use
            cache_manager: Cache manager instance
            config: Optional configuration parameters
        """
        self.device = device
        self.cache_manager = cache_manager
        self.config = config or {}
        
        # Initialize models
        self._initialize_models()
        
        # Setup processing pools with limited workers to manage memory better
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 2)
        )
        
        # Load custom vocabulary
        self.custom_vocab = self._load_custom_vocabulary()
        
        # Progress tracking
        self.progress_callback = None
        self.transcription_progress = 0.0
        
        logger.info(f"Transcriber initialized on {self.device}")

    def _initialize_models(self) -> None:
        """Initialize transcription models with proper error handling and memory optimization."""
        try:
            # Initialize Faster Whisper with memory optimization
            self.whisper = self._initialize_whisper()
            
            # Initialize Conformer model as backup
            self.conformer = self._initialize_conformer()
            
            # Initialize French-specific model if needed
            self.french_model = self._initialize_french_model() if self.config.get('optimize_for_french', True) else None
            
            logger.info("Transcription models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing transcription models: {str(e)}")
            raise TranscriptionException(f"Model initialization failed: {str(e)}")

    def _initialize_whisper(self) -> WhisperModel:
        """Initialize Whisper model with memory optimization."""
        try:
            whisper_model_size = self.config.get('whisper_model', 'large-v3')
            compute_type = "float16" if self.device.type == "cuda" else "int8"  # Use quantized for CPU
            
            # Use disk cache to reduce memory usage
            cache_dir = self.cache_manager.get_path("whisper_models")
            
            return WhisperModel(
                model_size_or_path=whisper_model_size,
                device=self.device.type,
                compute_type=compute_type,
                download_root=str(cache_dir),
                cpu_threads=self.config.get('cpu_threads', 4),
                num_workers=self.config.get('num_workers', 1)
            )
        except Exception as e:
            logger.error(f"Whisper initialization failed: {str(e)}")
            raise TranscriptionException(f"Whisper initialization failed: {str(e)}")

    def _initialize_conformer(self) -> Dict[str, Any]:
        """Initialize Conformer model for backup transcription."""
        try:
            model_id = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
            cache_dir = self.cache_manager.get_path("conformer")
            
            # Load processor first
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=str(cache_dir)
            )
            
            # Load model with memory optimizations
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=str(cache_dir)
            ).to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.warning(f"Conformer initialization failed: {str(e)}, will use fallback options")
            return None
            
    def _initialize_french_model(self) -> Dict[str, Any]:
        """Initialize French-specific model for enhanced French transcription."""
        try:
            # French-specific model (wav2vec2-large-xlsr-53-french)
            model_id = "facebook/wav2vec2-large-xlsr-53-french"
            cache_dir = self.cache_manager.get_path("french_asr")
            
            # Load processor first
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=str(cache_dir)
            )
            
            # Load model with memory optimizations
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=str(cache_dir)
            ).to(self.device)
            
            return {"model": model, "processor": processor}
        except Exception as e:
            logger.warning(f"French model initialization failed: {str(e)}, French optimization disabled")
            return None

    def _load_custom_vocabulary(self) -> Dict[str, Any]:
        """Load custom vocabulary for domain-specific terms."""
        vocab_path = self.config.get('custom_vocab_path')
        if not vocab_path:
            # Load default French vocabulary enhancements
            default_vocab = {
                'fr': {
                    # Common French mis-transcriptions and their corrections
                    "ces't": "c'est",
                    "s'est": "c'est",
                    "sa va": "ça va",
                    "sa c'est": "ça c'est",
                    "la bas": "là-bas",
                    "dejas": "déjà",
                    "tres": "très"
                },
                'en': {}
            }
            logger.info("Using default vocabulary enhancements")
            return default_vocab
            
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            logger.info(f"Loaded custom vocabulary with {sum(len(words) for words in vocab.values())} terms")
            return vocab
        except Exception as e:
            logger.warning(f"Failed to load custom vocabulary: {str(e)}")
            return {}

    async def transcribe(
        self,
        audio: Dict[str, Any],
        primary_language: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio with enhanced accuracy and memory optimization.
        
        Args:
            audio: Preprocessed audio data
            primary_language: Detected primary language
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            start_time = time.time()
            self.progress_callback = progress_callback
            self.transcription_progress = 0.0
            
            # Update progress
            await self._update_progress(0.0, "Starting transcription")
            
            # Choose optimal transcription approach based on language
            transcription_strategy = self._select_transcription_strategy(primary_language)
            
            # Prepare audio segments with intelligent chunking
            await self._update_progress(0.1, "Preparing audio segments")
            segments = await self._prepare_segments(audio['audio'], audio.get('sample_rate', 16000))
            
            # Update progress
            await self._update_progress(0.2, "Processing segments")
            
            # Process segments in batches to manage memory
            segment_results = await self._process_segments_in_batches(segments, primary_language, transcription_strategy)
            
            # Clean up memory after processing
            await self._manage_memory()
            
            # Update progress
            await self._update_progress(0.8, "Enhancing transcription")
            
            # Verify and enhance results
            enhanced_results = await self._enhance_transcription(
                segment_results,
                primary_language
            )
            
            # Format final results
            result = TranscriptionResult(
                segments=enhanced_results['segments'],
                text=enhanced_results['full_text'],
                confidence=enhanced_results['confidence'],
                language=primary_language,
                audio_quality=audio.get('quality_metrics', {}),
                metadata={
                    'model_used': enhanced_results['model_used'],
                    'processing_time': time.time() - start_time,
                    'segment_count': len(enhanced_results['segments']),
                    'language_optimized': primary_language == 'fr' and self.french_model is not None
                }
            )
            
            # Update progress
            await self._update_progress(1.0, "Transcription complete")
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise TranscriptionException(f"Transcription failed: {str(e)}")
        finally:
            # Clean up resources
            await self._manage_memory()

    async def _update_progress(self, progress: float, status: str) -> None:
        """Update transcription progress."""
        self.transcription_progress = progress
        if self.progress_callback:
            await self.progress_callback(progress, status)

    async def _manage_memory(self) -> None:
        """Manage memory to prevent OOM errors."""
        try:
            # Clear CUDA cache if using GPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                
            # Force garbage collection
            gc.collect()
            
            # Monitor memory usage
            if self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.debug(f"GPU memory in use: {memory_allocated:.2f} GB")
            else:
                memory_info = psutil.virtual_memory()
                logger.debug(f"System memory: {memory_info.percent}% used")
                
        except Exception as e:
            logger.warning(f"Memory management failed: {str(e)}")

    def _select_transcription_strategy(self, language: str) -> str:
        """Select optimal transcription strategy based on language."""
        # Optimize for French
        if language == 'fr' and self.french_model is not None:
            return 'french'
        
        # Default to whisper for most languages
        return 'whisper'

    async def _prepare_segments(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        min_segment_length: float = 2.0,
        max_segment_length: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Prepare audio segments with intelligent chunking based on silence detection."""
        loop = asyncio.get_event_loop()
        
        def _prepare():
            segments = []
            audio_np = audio.cpu().numpy()
            
            # Detect silence for smart segmentation
            try:
                # Detect non-silent sections
                non_silent_sections = librosa.effects.split(
                    audio_np,
                    top_db=self.config.get('silence_threshold', 20),
                    frame_length=2048,
                    hop_length=512
                )
                
                if len(non_silent_sections) == 0:
                    # No clear sections detected, fall back to time-based chunking
                    return self._fallback_segment_preparation(audio_np, sample_rate)
                
                # Process each non-silent section
                for start, end in non_silent_sections:
                    duration = (end - start) / sample_rate
                    
                    if duration < min_segment_length:
                        continue
                        
                    # Split long segments
                    if duration > max_segment_length:
                        num_splits = int(np.ceil(duration / max_segment_length))
                        split_size = (end - start) // num_splits
                        
                        for i in range(num_splits):
                            split_start = start + i * split_size
                            split_end = min(split_start + split_size, end)
                            
                            segments.append({
                                'audio': audio_np[split_start:split_end],
                                'start_time': split_start / sample_rate,
                                'end_time': split_end / sample_rate
                            })
                    else:
                        segments.append({
                            'audio': audio_np[start:end],
                            'start_time': start / sample_rate,
                            'end_time': end / sample_rate
                        })
                
                return segments
                
            except Exception as e:
                logger.warning(f"Smart segmentation failed: {str(e)}, falling back to basic chunking")
                return self._fallback_segment_preparation(audio_np, sample_rate)
        
        return await loop.run_in_executor(self.thread_pool, _prepare)

    def _fallback_segment_preparation(
        self,
        audio_np: np.ndarray,
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Fallback to simple time-based segmentation."""
        segments = []
        chunk_size = int(30.0 * sample_rate)  # 30 second chunks
        overlap = int(2.0 * sample_rate)  # 2 second overlap
        
        for start in range(0, len(audio_np), chunk_size - overlap):
            end = min(start + chunk_size, len(audio_np))
            
            segments.append({
                'audio': audio_np[start:end],
                'start_time': start / sample_rate,
                'end_time': end / sample_rate
            })
            
            if end == len(audio_np):
                break
                
        return segments

    async def _process_segments_in_batches(
        self,
        segments: List[Dict[str, Any]],
        language: str,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Process segments in batches to manage memory usage."""
        # Determine batch size based on device
        batch_size = 3 if self.device.type == "cuda" else 1
        segment_results = []
        
        total_segments = len(segments)
        for batch_idx in range(0, total_segments, batch_size):
            # Process batch
            batch = segments[batch_idx:min(batch_idx + batch_size, total_segments)]
            
            # Update progress
            progress = 0.2 + 0.6 * (batch_idx / total_segments)
            await self._update_progress(
                progress,
                f"Transcribing segments {batch_idx+1}-{min(batch_idx+batch_size, total_segments)} of {total_segments}"
            )
            
            # Process segments in parallel
            batch_tasks = [
                self._process_segment(segment, language, strategy)
                for segment in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            segment_results.extend(batch_results)
            
            # Clean up memory after each batch
            await self._manage_memory()
            
        return segment_results

    async def _process_segment(
        self,
        segment: Dict[str, Any],
        language: str,
        strategy: str
    ) -> Dict[str, Any]:
        """Process individual audio segment with fallback options."""
        loop = asyncio.get_event_loop()
        
        # Generate unique ID for this segment
        segment_id = f"{int(segment['start_time'] * 1000)}-{int(segment['end_time'] * 1000)}"
        
        # Choose processing method based on strategy and language
        if strategy == 'french' and language == 'fr':
            try:
                french_result = await loop.run_in_executor(
                    self.thread_pool,
                    self._transcribe_with_french_model,
                    segment['audio']
                )
                
                # Use French model if confidence is high enough
                if french_result['confidence'] >= 0.7:
                    return {
                        **french_result,
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'id': segment_id,
                        'model_used': 'french_wav2vec',
                        'language': language
                    }
            except Exception as e:
                logger.warning(f"French transcription failed: {str(e)}")
                
        # Try Whisper
        try:
            whisper_result = await loop.run_in_executor(
                self.thread_pool,
                self._transcribe_with_whisper,
                segment['audio'],
                language
            )
            
            return {
                **whisper_result,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'id': segment_id,
                'model_used': 'whisper',
                'language': language
            }
                
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {str(e)}")
            
        # Fallback to Conformer if available
        if self.conformer:
            try:
                conformer_result = await loop.run_in_executor(
                    self.thread_pool,
                    self._transcribe_with_conformer,
                    segment['audio']
                )
                
                return {
                    **conformer_result,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'id': segment_id,
                    'model_used': 'conformer',
                    'language': language
                }
                
            except Exception as e:
                logger.error(f"All transcription models failed: {str(e)}")
        
        # Emergency fallback with empty result
        logger.error(f"All transcription methods failed for segment {segment_id}")
        return {
            'text': '',
            'segments': [],
            'confidence': 0.0,
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'id': segment_id,
            'model_used': 'failed',
            'language': language
        }

    def _transcribe_with_whisper(
        self,
        audio: np.ndarray,
        language: str
    ) -> Dict[str, Any]:
        """Transcribe using Faster Whisper with optimizations."""
        segments, info = self.whisper.transcribe(
            audio,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Optimize for natural speech
                speech_pad_ms=150,
                threshold=0.35  # Slightly more sensitive
            )
        )
        
        text_segments = []
        for segment in segments:
            text_segments.append({
                'text': segment.text,
                'words': segment.words,
                'confidence': segment.avg_logprob
            })
            
        # Calculate overall confidence
        if text_segments:
            avg_confidence = np.mean([seg['confidence'] for seg in text_segments])
            # Convert to probability (0-1) from log probability
            confidence = min(max(np.exp(avg_confidence), 0.0), 1.0)
        else:
            confidence = 0.0
            
        return {
            'text': ' '.join(seg['text'] for seg in text_segments),
            'segments': text_segments,
            'confidence': confidence
        }

    def _transcribe_with_conformer(
        self,
        audio: np.ndarray
    ) -> Dict[str, Any]:
        """Transcribe using Conformer model as backup."""
        inputs = self.conformer["processor"](
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.conformer["model"].generate(
                inputs.input_values,
                max_length=self.config.get('max_length', 448)
            )
            
        transcription = self.conformer["processor"].batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return {
            'text': transcription[0],
            'segments': [{'text': transcription[0], 'confidence': 0.8}],
            'confidence': 0.8  # Conformer doesn't provide detailed confidence scores
        }

    def _transcribe_with_french_model(
        self,
        audio: np.ndarray
    ) -> Dict[str, Any]:
        """Transcribe using specialized French model."""
        inputs = self.french_model["processor"](
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.french_model["model"](inputs.input_values)
            logits = outputs.logits
            
            # Calculate confidence based on softmax probabilities
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1)[0].mean().item()
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.french_model["processor"].batch_decode(predicted_ids)[0]
        
        return {
            'text': transcription,
            'segments': [{'text': transcription, 'confidence': confidence}],
            'confidence': confidence
        }

    async def _enhance_transcription(
        self,
        segment_results: List[Dict[str, Any]],
        language: str
    ) -> Dict[str, Any]:
        """Enhance transcription results with post-processing and custom vocabulary."""
        start_time = time.time()
        
        # Combine segments
        combined_segments = []
        full_text = []
        confidences = []
        models_used = {}
        
        for segment in segment_results:
            # Skip empty segments
            if not segment['text'].strip():
                continue
                
            # Apply custom vocabulary
            enhanced_text = await self._apply_custom_vocabulary(
                segment['text'],
                language
            )
            
            # Add to combined results
            combined_segments.append({
                'id': segment['id'],
                'text': enhanced_text,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'confidence': segment['confidence'],
                'model_used': segment['model_used'],
                'language': segment['language']
            })
            
            full_text.append(enhanced_text)
            confidences.append(segment['confidence'])
            models_used[segment['model_used']] = models_used.get(segment['model_used'], 0) + 1
            
        # Determine primary model used
        primary_model = max(models_used.items(), key=lambda x: x[1])[0] if models_used else "unknown"
        
        # Fill gaps in timestamps if necessary
        filled_segments = self._fill_timestamp_gaps(combined_segments)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'segments': filled_segments,
            'full_text': ' '.join(full_text),
            'confidence': avg_confidence,
            'model_used': primary_model,
            'processing_time': time.time() - start_time
        }

    async def _apply_custom_vocabulary(
        self,
        text: str,
        language: str
    ) -> str:
        """Apply custom vocabulary corrections with async processing for large texts."""
        if not self.custom_vocab or language not in self.custom_vocab:
            return text
            
        # For short text, process directly
        if len(text) < 1000:
            enhanced_text = text
            for term, replacement in self.custom_vocab[language].items():
                enhanced_text = enhanced_text.replace(term, replacement)
            return enhanced_text
            
        # For longer text, process asynchronously in chunks
        loop = asyncio.get_event_loop()
        
        def _process_chunk(chunk: str) -> str:
            enhanced = chunk
            for term, replacement in self.custom_vocab[language].items():
                enhanced = enhanced.replace(term, replacement)
            return enhanced
            
        # Split into chunks
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Process chunks concurrently
        tasks = [loop.run_in_executor(None, _process_chunk, chunk) for chunk in chunks]
        processed_chunks = await asyncio.gather(*tasks)
        
        return ''.join(processed_chunks)

    def _fill_timestamp_gaps(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fill gaps in timestamps for consistent timeline."""
        if not segments:
            return []
            
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        
        # Check and fill gaps
        for i in range(len(sorted_segments) - 1):
            current = sorted_segments[i]
            next_seg = sorted_segments[i + 1]
            
            # If there's a gap and it's not too large
            gap = next_seg['start_time'] - current['end_time']
            if 0 < gap < 2.0:  # Small gaps (less than 2 seconds)
                # Adjust ending of current segment to match start of next
                midpoint = (current['end_time'] + next_seg['start_time']) / 2
                current['end_time'] = midpoint
                next_seg['start_time'] = midpoint
                
        return sorted_segments

    async def cleanup(self):
        """Cleanup resources and free memory."""
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clean up model references to help garbage collection
            self.french_model = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            gc.collect()
            
            logger.info("Transcriber resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during transcriber cleanup: {str(e)}")