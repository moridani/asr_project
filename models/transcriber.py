from typing import Dict, Any, List, Optional, Union
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

class Transcriber:
    def __init__(
        self,
        device: torch.device,
        cache_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize transcription models with optimizations.
        
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
        
        # Setup processing pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 3)
        )
        
        # Load custom vocabulary if provided
        self.custom_vocab = self._load_custom_vocabulary()
        
        logger.info("Transcriber initialized successfully")

    def _initialize_models(self) -> None:
        """Initialize transcription models with proper error handling."""
        try:
            # Initialize Faster Whisper
            self.whisper = WhisperModel(
                model_size_or_path=self.config.get('whisper_model', 'large-v3'),
                device=self.device.type,
                compute_type="float16" if self.device.type == "cuda" else "float32",
                cpu_threads=6,
                num_workers=2
            )
            
            # Initialize Conformer model as backup
            self.conformer = self._initialize_conformer()
            
            logger.info("Transcription models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing transcription models: {str(e)}")
            raise

    def _initialize_conformer(self) -> Pipeline:
        """Initialize Conformer model for backup transcription."""
        model_id = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        return {"model": model, "processor": processor}

    def _load_custom_vocabulary(self) -> Dict[str, Any]:
        """Load custom vocabulary for domain-specific terms."""
        vocab_path = self.config.get('custom_vocab_path')
        if not vocab_path:
            return {}
            
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            logger.info(f"Loaded custom vocabulary with {len(vocab)} terms")
            return vocab
        except Exception as e:
            logger.warning(f"Failed to load custom vocabulary: {str(e)}")
            return {}

    async def transcribe(
        self,
        audio: Dict[str, Any],
        primary_language: str
    ) -> Dict[str, Any]:
        """
        Transcribe audio with enhanced accuracy and error handling.
        
        Args:
            audio: Preprocessed audio data
            primary_language: Detected primary language
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Prepare audio segments
            segments = await self._prepare_segments(audio['audio'])
            
            # Process segments in parallel
            transcription_tasks = [
                self._process_segment(segment, primary_language)
                for segment in segments
            ]
            
            segment_results = await asyncio.gather(*transcription_tasks)
            
            # Verify and enhance results
            enhanced_results = await self._enhance_transcription(
                segment_results,
                primary_language
            )
            
            # Format final results
            return {
                'segments': enhanced_results['segments'],
                'text': enhanced_results['full_text'],
                'confidence': enhanced_results['confidence'],
                'language': primary_language,
                'audio_quality': audio.get('quality_metrics', {}),
                'metadata': {
                    'model_used': enhanced_results['model_used'],
                    'processing_time': enhanced_results['processing_time'],
                    'segment_count': len(enhanced_results['segments'])
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    async def _prepare_segments(
        self,
        audio: torch.Tensor,
        min_segment_length: float = 2.0,
        max_segment_length: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Prepare audio segments for processing."""
        segments = []
        audio_np = audio.cpu().numpy()
        
        # Detect silence for smart segmentation
        loop = asyncio.get_event_loop()
        
        def _detect_silence():
            return librosa.effects.split(
                audio_np,
                top_db=self.config.get('silence_threshold', 20),
                frame_length=2048,
                hop_length=512
            )
            
        silence_boundaries = await loop.run_in_executor(
            self.thread_pool,
            _detect_silence
        )
        
        # Create segments based on silence
        for start, end in silence_boundaries:
            duration = (end - start) / self.config.get('sample_rate', 16000)
            
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
                        'start_time': split_start / self.config.get('sample_rate', 16000),
                        'end_time': split_end / self.config.get('sample_rate', 16000)
                    })
            else:
                segments.append({
                    'audio': audio_np[start:end],
                    'start_time': start / self.config.get('sample_rate', 16000),
                    'end_time': end / self.config.get('sample_rate', 16000)
                })
                
        return segments

    async def _process_segment(
        self,
        segment: Dict[str, Any],
        language: str
    ) -> Dict[str, Any]:
        """Process individual audio segment with fallback options."""
        loop = asyncio.get_event_loop()
        
        # Try Whisper first
        try:
            whisper_result = await loop.run_in_executor(
                self.thread_pool,
                self._transcribe_with_whisper,
                segment['audio'],
                language
            )
            
            if whisper_result['confidence'] >= 0.8:
                return {
                    **whisper_result,
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'model_used': 'whisper'
                }
                
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {str(e)}")
            
        # Fallback to Conformer
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
                'model_used': 'conformer'
            }
            
        except Exception as e:
            logger.error(f"Both transcription models failed: {str(e)}")
            raise

    def _transcribe_with_whisper(
        self,
        audio: np.ndarray,
        language: str
    ) -> Dict[str, Any]:
        """Transcribe using Faster Whisper."""
        segments, info = self.whisper.transcribe(
            audio,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )
        
        text_segments = []
        for segment in segments:
            text_segments.append({
                'text': segment.text,
                'words': segment.words,
                'confidence': segment.avg_logprob
            })
            
        return {
            'text': ' '.join(seg['text'] for seg in text_segments),
            'segments': text_segments,
            'confidence': np.mean([seg['confidence'] for seg in text_segments])
        }

    def _transcribe_with_conformer(
        self,
        audio: np.ndarray
    ) -> Dict[str, Any]:
        """Transcribe using Conformer model."""
        inputs = self.conformer["processor"](
            audio,
            sampling_rate=self.config.get('sample_rate', 16000),
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
            'confidence': 0.8  # Conformer doesn't provide confidence scores
        }

    async def _enhance_transcription(
        self,
        segment_results: List[Dict[str, Any]],
        language: str
    ) -> Dict[str, Any]:
        """Enhance transcription results with post-processing."""
        start_time = time.time()
        
        # Combine segments
        combined_segments = []
        full_text = []
        confidences = []
        
        for segment in segment_results:
            # Apply custom vocabulary
            enhanced_text = self._apply_custom_vocabulary(
                segment['text'],
                language
            )
            
            combined_segments.append({
                'text': enhanced_text,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'confidence': segment['confidence'],
                'model_used': segment['model_used']
            })
            
            full_text.append(enhanced_text)
            confidences.append(segment['confidence'])
            
        return {
            'segments': combined_segments,
            'full_text': ' '.join(full_text),
            'confidence': np.mean(confidences),
            'model_used': self._determine_primary_model(segment_results),
            'processing_time': time.time() - start_time
        }

    def _apply_custom_vocabulary(
        self,
        text: str,
        language: str
    ) -> str:
        """Apply custom vocabulary corrections."""
        if not self.custom_vocab or language not in self.custom_vocab:
            return text
            
        enhanced_text = text
        for term, replacement in self.custom_vocab[language].items():
            enhanced_text = enhanced_text.replace(term, replacement)
            
        return enhanced_text

    @staticmethod
    def _determine_primary_model(
        segment_results: List[Dict[str, Any]]
    ) -> str:
        """Determine which model was primarily used."""
        model_counts = {}
        for segment in segment_results:
            model = segment['model_used']
            model_counts[model] = model_counts.get(model, 0) + 1
            
        return max(model_counts.items(), key=lambda x: x[1])[0]