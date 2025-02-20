from typing import Dict, Any, List, Optional
import torch
import asyncio
from loguru import logger
from pathlib import Path
import time
from rich.progress import Progress
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

from models.language_detector import LanguageDetector
from models.transcriber import Transcriber
from models.diarizer import Diarizer
from models.translator import Translator
from utils.audio_processor import AudioProcessor
from utils.error_handler import handle_error, ASRException
from utils.validators import validate_audio_file
from utils.cache_manager import ModelCache

class ASRPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ASR pipeline with all required models.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self._setup_environment(config)
        self._initialize_components()
        logger.info(f"ASR Pipeline initialized on device: {self.device}")

    def _setup_environment(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup processing environment and configurations."""
        self.config = config or {}
        
        # Setup device with optimal settings
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            # Set optimal CUDA settings
            torch.cuda.set_device(0)  # Use primary GPU
            ort.set_default_logger_severity(3)  # Reduce ONNX logging
        else:
            self.device = torch.device("cpu")
            # Optimize CPU settings
            torch.set_num_threads(self.config.get('cpu_threads', 4))

        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 3)
        )
        
        # Initialize model cache
        self.cache_manager = ModelCache(
            cache_dir=self.config.get('cache_dir', 'cache'),
            max_size_gb=self.config.get('max_cache_size', 10)
        )

    def _initialize_components(self) -> None:
        """Initialize pipeline components with error handling."""
        try:
            self.audio_processor = AudioProcessor(
                device=self.device,
                sample_rate=self.config.get('sample_rate', 16000)
            )
            
            self.language_detector = LanguageDetector(
                device=self.device,
                cache_manager=self.cache_manager
            )
            
            self.transcriber = Transcriber(
                device=self.device,
                cache_manager=self.cache_manager
            )
            
            self.diarizer = Diarizer(
                device=self.device,
                cache_manager=self.cache_manager
            )
            
            self.translator = Translator(
                device=self.device,
                cache_manager=self.cache_manager
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise ASRException("Pipeline initialization failed") from e

    async def process_file(
        self,
        file_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file through the complete pipeline with progress tracking.
        
        Args:
            file_path: Path to the audio file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processed results
        """
        start_time = time.time()
        try:
            # Validate input file
            file_path = validate_audio_file(file_path)
            
            # Create progress tracking
            progress_total = 5  # Total number of major steps
            current_progress = 0
            
            async def update_progress(step_name: str):
                nonlocal current_progress
                current_progress += 1
                if progress_callback:
                    await progress_callback(
                        current_progress / progress_total,
                        step_name
                    )
            
            # Process audio file with noise reduction
            audio = await self.audio_processor.load_and_preprocess(file_path)
            await update_progress("Audio Processing")
            
            # Parallel processing of independent tasks
            detection_task = asyncio.create_task(
                self.language_detector.detect(audio)
            )
            diarization_task = asyncio.create_task(
                self.diarizer.process(audio)
            )
            
            # Wait for parallel tasks
            language_info, diarization_result = await asyncio.gather(
                detection_task,
                diarization_task
            )
            await update_progress("Language Detection & Diarization")
            
            # Transcribe audio with detected language
            transcription = await self.transcriber.transcribe(
                audio, 
                language_info['primary_language']
            )
            await update_progress("Transcription")
            
            # Translate non-English segments
            translations = await self.translator.translate_segments(
                transcription,
                diarization_result
            )
            await update_progress("Translation")
            
            # Combine results
            result = self._combine_results(
                transcription,
                diarization_result,
                translations,
                start_time
            )
            await update_progress("Finalizing")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return handle_error(e, file_path)
        finally:
            # Cleanup and release resources
            await self._cleanup()

    def _combine_results(
        self,
        transcription: Dict[str, Any],
        diarization: Dict[str, Any],
        translations: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Combine results from different components into a unified format."""
        processing_time = time.time() - start_time
        
        combined_segments = []
        for segment in transcription['segments']:
            speaker_info = self._find_speaker(
                segment['start'],
                segment['end'],
                diarization['segments']
            )
            
            combined_segments.append({
                'start_time': segment['start'],
                'end_time': segment['end'],
                'speaker': speaker_info['speaker_id'],
                'speaker_confidence': speaker_info['confidence'],
                'text': segment['text'],
                'translation': translations.get(segment['id']),
                'language': segment['language'],
                'confidence': segment['confidence']
            })
        
        return {
            'language_distribution': self._format_language_distribution(
                transcription['language_info']
            ),
            'speakers': diarization['speakers'],
            'segments': combined_segments,
            'metadata': {
                'processing_time': processing_time,
                'audio_quality': transcription['audio_quality'],
                'confidence_scores': {
                    'language_detection': transcription['language_info']['confidence'],
                    'transcription': transcription['confidence'],
                    'diarization': diarization['confidence']
                }
            }
        }

    @staticmethod
    def _find_speaker(
        start: float,
        end: float,
        diarization_segments: List[Dict]
    ) -> Dict[str, Any]:
        """Find the speaker for a given time segment with confidence score."""
        max_overlap = 0
        best_speaker = {'speaker_id': 'unknown', 'confidence': 0.0}
        
        for segment in diarization_segments:
            overlap_start = max(start, segment['start'])
            overlap_end = min(end, segment['end'])
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = {
                        'speaker_id': segment['speaker_id'],
                        'confidence': segment['confidence']
                    }
        
        return best_speaker

    async def _cleanup(self) -> None:
        """Cleanup resources after processing."""
        torch.cuda.empty_cache()
        await self.cache_manager.cleanup()

    @staticmethod
    def _format_language_distribution(
        language_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Format language distribution information."""
        return [
            {
                'language': lang['language'],
                'percentage': round(lang['confidence'] * 100, 2),
                'confidence': lang['confidence']
            }
            for lang in language_info['distribution'][:3]  # Top 3 languages
        ]