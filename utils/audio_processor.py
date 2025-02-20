from typing import Dict, Any, Optional, Tuple
import torch
import torchaudio
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AudioProcessor:
    def __init__(
        self,
        device: torch.device,
        sample_rate: int = 16000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audio processor with optimized settings.
        
        Args:
            device: torch device to use
            sample_rate: target sample rate
            config: optional configuration dictionary
        """
        self.device = device
        self.sample_rate = sample_rate
        self.config = config or {}
        
        # Initialize audio processing settings
        self.chunk_size = self.config.get('chunk_size', 30)  # seconds
        self.overlap = self.config.get('overlap', 0.5)  # 50% overlap
        self.min_duration = self.config.get('min_duration', 0.1)  # seconds
        
        # Setup processing pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 3)
        )

    async def load_and_preprocess(self, file_path: str) -> Dict[str, Any]:
        """
        Load and preprocess audio file with optimizations.
        
        Args:
            file_path: path to audio file
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        try:
            # Load audio file
            audio_data = await self._load_audio(file_path)
            
            # Process in chunks for memory efficiency
            processed_chunks = await self._process_chunks(audio_data)
            
            # Combine chunks and compute quality metrics
            final_audio = self._combine_chunks(processed_chunks)
            quality_metrics = await self._compute_quality_metrics(final_audio)
            
            return {
                'audio': final_audio['audio'],
                'sample_rate': self.sample_rate,
                'duration': final_audio['duration'],
                'quality_metrics': quality_metrics,
                'metadata': final_audio['metadata']
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise

    async def _load_audio(self, file_path: str) -> Dict[str, Any]:
        """Load audio file with format detection and validation."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Handle different formats
            if path.suffix.lower() in ['.mp3', '.m4a', '.wma']:
                return await self._load_with_pydub(path)
            else:
                return await self._load_with_torchaudio(path)
                
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise

    async def _load_with_pydub(self, path: Path) -> Dict[str, Any]:
        """Load audio using pydub for format compatibility."""
        loop = asyncio.get_event_loop()
        
        def _load():
            audio = AudioSegment.from_file(str(path))
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                samples = samples.mean(axis=1)  # Convert to mono
                
            # Convert to float32
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            
            return {
                'audio': torch.from_numpy(samples).to(self.device),
                'sample_rate': audio.frame_rate,
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'metadata': {
                    'original_format': path.suffix,
                    'channels': audio.channels,
                    'frame_width': audio.sample_width
                }
            }
            
        return await loop.run_in_executor(self.thread_pool, _load)

    async def _load_with_torchaudio(self, path: Path) -> Dict[str, Any]:
        """Load audio using torchaudio for better performance."""
        loop = asyncio.get_event_loop()
        
        def _load():
            waveform, sample_rate = torchaudio.load(str(path))
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze()
            
            metadata = torchaudio.info(str(path))
            
            return {
                'audio': waveform.to(self.device),
                'sample_rate': sample_rate,
                'duration': waveform.shape[-1] / sample_rate,
                'metadata': {
                    'original_format': path.suffix,
                    'channels': metadata.num_channels,
                    'bits_per_sample': getattr(metadata, 'bits_per_sample', None)
                }
            }
            
        return await loop.run_in_executor(self.thread_pool, _load)

    async def _process_chunks(self, audio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process audio in chunks for memory efficiency."""
        audio = audio_data['audio']
        chunk_samples = int(self.chunk_size * self.sample_rate)
        overlap_samples = int(self.overlap * chunk_samples)
        
        chunks = []
        for start in range(0, len(audio), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            if len(chunk) < self.min_duration * self.sample_rate:
                continue
                
            # Process chunk
            processed_chunk = await self._process_single_chunk(chunk)
            chunks.append({
                'audio': processed_chunk,
                'start': start / self.sample_rate,
                'end': end / self.sample_rate
            })
            
        return chunks

    async def _process_single_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Process a single audio chunk with optimizations."""
        loop = asyncio.get_event_loop()
        
        def _process():
            # Convert to numpy for processing
            audio_np = chunk.cpu().numpy()
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_np,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.75
            )
            
            # Normalize audio
            normalized = librosa.util.normalize(reduced_noise)
            
            # Convert back to torch tensor
            return torch.from_numpy(normalized).to(self.device)
            
        return await loop.run_in_executor(self.thread_pool, _process)

    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine processed chunks with overlap handling."""
        if not chunks:
            raise ValueError("No audio chunks to combine")
            
        # Calculate total length
        total_length = int(chunks[-1]['end'] * self.sample_rate)
        combined = torch.zeros(total_length, device=self.device)
        
        # Overlap-add chunks
        for chunk in chunks:
            start_idx = int(chunk['start'] * self.sample_rate)
            end_idx = int(chunk['end'] * self.sample_rate)
            
            # Apply fade for smooth transitions
            fade_len = int(self.overlap * self.sample_rate)
            if start_idx > 0:
                chunk['audio'][:fade_len] *= torch.linspace(0, 1, fade_len, device=self.device)
            if end_idx < total_length:
                chunk['audio'][-fade_len:] *= torch.linspace(1, 0, fade_len, device=self.device)
                
            combined[start_idx:end_idx] += chunk['audio']
            
        return {
            'audio': combined,
            'duration': total_length / self.sample_rate,
            'metadata': {
                'processed': True,
                'chunks': len(chunks),
                'sample_rate': self.sample_rate
            }
        }

    async def _compute_quality_metrics(self, audio_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute audio quality metrics."""
        loop = asyncio.get_event_loop()
        
        def _compute():
            audio = audio_data['audio'].cpu().numpy()
            
            metrics = {
                'rms_energy': float(np.sqrt(np.mean(audio ** 2))),
                'peak_amplitude': float(np.max(np.abs(audio))),
                'signal_to_noise': float(self._estimate_snr(audio)),
                'duration': audio_data['duration']
            }
            
            # Add spectral metrics
            if len(audio) > self.sample_rate:  # At least 1 second
                spectral = librosa.stft(audio)
                metrics.update({
                    'spectral_flatness': float(np.mean(librosa.feature.spectral_flatness(S=np.abs(spectral)))),
                    'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(S=np.abs(spectral))))
                })
                
            return metrics
            
        return await loop.run_in_executor(self.thread_pool, _compute)

    @staticmethod
    def _estimate_snr(audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        noise_floor = np.sort(np.abs(audio))[int(len(audio) * 0.1)]  # 10th percentile
        signal_power = np.mean(audio ** 2)
        noise_power = noise_floor ** 2
        
        if noise_power == 0:
            return 100.0  # High SNR default
            
        snr = 10 * np.log10(signal_power / noise_power)
        return min(max(snr, 0), 100)  # Clip to reasonable range