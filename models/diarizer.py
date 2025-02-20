from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline, Annotation
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

class Diarizer:
    def __init__(
        self,
        device: torch.device,
        cache_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize speaker diarization system with multiple models for robustness.
        
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
        
        # Configuration parameters
        self.min_speakers = self.config.get('min_speakers', 1)
        self.max_speakers = self.config.get('max_speakers', 10)
        self.threshold = self.config.get('threshold', 0.5)
        
        logger.info("Diarizer initialized successfully")

    def _initialize_models(self) -> None:
        """Initialize diarization models with proper error handling."""
        try:
            # Initialize PyAnnote pipeline
            self.pyannote = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.get('hf_token')
            ).to(self.device)
            
            # Initialize ECAPA-TDNN for speaker embeddings
            self.ecapa_tdnn = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.cache_manager.get_path("ecapa_tdnn")
            ).to(self.device)
            
            # Initialize Resemblyzer for verification
            self.resemblyzer = VoiceEncoder(
                device=self.device.type,
                weights_fpath=None  # Use default weights
            )
            
            logger.info("Diarization models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing diarization models: {str(e)}")
            raise

    async def process(
        self,
        audio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process audio for speaker diarization with enhanced accuracy.
        
        Args:
            audio: Preprocessed audio data
            
        Returns:
            Dictionary containing diarization results
        """
        try:
            start_time = time.time()
            
            # Initial diarization with PyAnnote
            pyannote_results = await self._pyannote_diarization(audio)
            
            # Extract speaker embeddings
            embeddings = await self._extract_speaker_embeddings(
                audio,
                pyannote_results['segments']
            )
            
            # Refine speaker clustering
            refined_results = await self._refine_speaker_clusters(
                pyannote_results['segments'],
                embeddings
            )
            
            # Detect overlapping speech
            overlap_regions = await self._detect_speech_overlap(
                audio,
                refined_results['segments']
            )
            
            # Combine and enhance results
            final_results = self._combine_results(
                refined_results,
                overlap_regions,
                time.time() - start_time
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise

    async def _pyannote_diarization(
        self,
        audio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform initial diarization using PyAnnote."""
        loop = asyncio.get_event_loop()
        
        def _diarize():
            # Convert audio to PyAnnote format
            waveform = torch.tensor(audio['audio']).unsqueeze(0)
            
            # Run diarization
            diarization = self.pyannote(
                {"waveform": waveform, "sample_rate": audio['sample_rate']},
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker_id': speaker,
                    'confidence': 0.0  # Will be updated later
                })
            
            return {
                'segments': segments,
                'num_speakers': len(set(s['speaker_id'] for s in segments))
            }
        
        return await loop.run_in_executor(self.thread_pool, _diarize)

    async def _extract_speaker_embeddings(
        self,
        audio: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Extract speaker embeddings using multiple models."""
        embeddings = {}
        
        async def process_segment(segment: Dict[str, Any]) -> Tuple[str, Dict[str, torch.Tensor]]:
            start_sample = int(segment['start'] * audio['sample_rate'])
            end_sample = int(segment['end'] * audio['sample_rate'])
            segment_audio = audio['audio'][start_sample:end_sample]
            
            # Get embeddings from both models
            ecapa_emb = await self._get_ecapa_embedding(segment_audio)
            resem_emb = await self._get_resemblyzer_embedding(segment_audio)
            
            return segment['speaker_id'], {
                'ecapa': ecapa_emb,
                'resemblyzer': resem_emb
            }
        
        # Process segments in parallel
        tasks = [process_segment(segment) for segment in segments]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        for speaker_id, embs in results:
            if speaker_id not in embeddings:
                embeddings[speaker_id] = {
                    'ecapa': [],
                    'resemblyzer': []
                }
            embeddings[speaker_id]['ecapa'].append(embs['ecapa'])
            embeddings[speaker_id]['resemblyzer'].append(embs['resemblyzer'])
        
        # Average embeddings for each speaker
        for speaker_id in embeddings:
            embeddings[speaker_id] = {
                'ecapa': torch.stack(embeddings[speaker_id]['ecapa']).mean(0),
                'resemblyzer': torch.stack(embeddings[speaker_id]['resemblyzer']).mean(0)
            }
        
        return embeddings

    async def _get_ecapa_embedding(
        self,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """Get ECAPA-TDNN embedding."""
        loop = asyncio.get_event_loop()
        
        def _extract():
            with torch.no_grad():
                return self.ecapa_tdnn.encode_batch(audio).squeeze()
        
        return await loop.run_in_executor(self.thread_pool, _extract)

    async def _get_resemblyzer_embedding(
        self,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """Get Resemblyzer embedding."""
        loop = asyncio.get_event_loop()
        
        def _extract():
            audio_np = audio.cpu().numpy()
            processed = preprocess_wav(audio_np)
            return torch.from_numpy(
                self.resemblyzer.embed_utterance(processed)
            ).to(self.device)
        
        return await loop.run_in_executor(self.thread_pool, _extract)

    async def _refine_speaker_clusters(
        self,
        segments: List[Dict[str, Any]],
        embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Refine speaker clusters using embeddings."""
        # Prepare distance matrix
        speakers = list(embeddings.keys())
        n_speakers = len(speakers)
        
        distance_matrix = torch.zeros((n_speakers, n_speakers))
        
        for i, spk1 in enumerate(speakers):
            for j, spk2 in enumerate(speakers):
                if i < j:
                    # Compute combined distance using both embeddings
                    ecapa_dist = F.cosine_similarity(
                        embeddings[spk1]['ecapa'].unsqueeze(0),
                        embeddings[spk2]['ecapa'].unsqueeze(0)
                    )
                    resem_dist = F.cosine_similarity(
                        embeddings[spk1]['resemblyzer'].unsqueeze(0),
                        embeddings[spk2]['resemblyzer'].unsqueeze(0)
                    )
                    
                    # Weighted combination
                    combined_dist = 0.7 * ecapa_dist + 0.3 * resem_dist
                    distance_matrix[i, j] = distance_matrix[j, i] = combined_dist
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            affinity='precomputed',
            linkage='complete'
        )
        
        labels = clustering.fit_predict(distance_matrix.cpu().numpy())
        
        # Update speaker IDs and compute confidences
        new_segments = []
        for segment in segments:
            old_speaker_idx = speakers.index(segment['speaker_id'])
            new_speaker_id = f"speaker_{labels[old_speaker_idx]}"
            
            # Compute confidence based on embedding similarities
            confidence = self._compute_speaker_confidence(
                embeddings[segment['speaker_id']],
                labels[old_speaker_idx],
                embeddings,
                labels
            )
            
            new_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'speaker_id': new_speaker_id,
                'confidence': confidence
            })
        
        return {
            'segments': new_segments,
            'num_speakers': len(set(labels))
        }

    def _compute_speaker_confidence(
        self,
        speaker_emb: Dict[str, torch.Tensor],
        speaker_label: int,
        all_embeddings: Dict[str, Dict[str, torch.Tensor]],
        all_labels: np.ndarray
    ) -> float:
        """Compute confidence score for speaker assignment."""
        # Get all embeddings for the same cluster
        cluster_embeddings = {
            'ecapa': [],
            'resemblyzer': []
        }
        
        for spk_id, label in zip(all_embeddings.keys(), all_labels):
            if label == speaker_label:
                cluster_embeddings['ecapa'].append(all_embeddings[spk_id]['ecapa'])
                cluster_embeddings['resemblyzer'].append(all_embeddings[spk_id]['resemblyzer'])
        
        # Compute average similarity to cluster
        ecapa_sims = torch.stack([
            F.cosine_similarity(speaker_emb['ecapa'], emb, dim=0)
            for emb in cluster_embeddings['ecapa']
        ])
        
        resem_sims = torch.stack([
            F.cosine_similarity(speaker_emb['resemblyzer'], emb, dim=0)
            for emb in cluster_embeddings['resemblyzer']
        ])
        
        # Weighted combination of similarities
        confidence = (0.7 * ecapa_sims.mean() + 0.3 * resem_sims.mean()).item()
        
        return min(max(confidence, 0.0), 1.0)

    async def _detect_speech_overlap(
        self,
        audio: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect regions of overlapping speech."""
        loop = asyncio.get_event_loop()
        
        def _detect():
            overlap_regions = []
            
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x['start'])
            
            # Check for overlaps
            for i in range(len(sorted_segments) - 1):
                current = sorted_segments[i]
                next_seg = sorted_segments[i + 1]
                
                if current['end'] > next_seg['start']:
                    overlap_regions.append({
                        'start': next_seg['start'],
                        'end': min(current['end'], next_seg['end']),
                        'speakers': [current['speaker_id'], next_seg['speaker_id']],
                        'confidence': min(current['confidence'], next_seg['confidence'])
                    })
            
            return overlap_regions
        
        return await loop.run_in_executor(self.thread_pool, _detect)

    def _combine_results(
        self,
        diarization_results: Dict[str, Any],
        overlap_regions: List[Dict[str, Any]],
        processing_time: float
    ) -> Dict[str, Any]:
        """Combine and format final results."""
        # Get unique speakers and their segments
        speakers = {}
        for segment in diarization_results['segments']:
            if segment['speaker_id'] not in speakers:
                speakers[segment['speaker_id']] = {
                    'total_duration': 0.0,
                    'segments': [],
                    'avg_confidence': 0.0
                }
            
            duration = segment['end'] - segment['start']
            speakers[segment['speaker_id']]['total_duration'] += duration
            speakers[segment['speaker_id']]['segments'].append(segment)
            speakers[segment['speaker_id']]['avg_confidence'] += (
                segment['confidence'] * duration
            )
        
        # Compute average confidence for each speaker
        for speaker_id in speakers:
            total_duration = speakers[speaker_id]['total_duration']
            speakers[speaker_id]['avg_confidence'] /= total_duration
        
        return {
            'segments': diarization_results['segments'],
            'overlap_regions': overlap_regions,
            'speakers': {
                speaker_id: {
                    'total_duration': info['total_duration'],
                    'avg_confidence': info['avg_confidence'],
                    'segment_count': len(info['segments'])
                }
                for speaker_id, info in speakers.items()
            },
            'num_speakers': diarization_results['num_speakers'],
            'metadata': {
                'processing_time': processing_time,
                'overlap_count': len(overlap_regions)
            }
        }