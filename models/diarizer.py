from typing import Dict, Any, List, Optional, Tuple, Callable, Union
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
import gc
import psutil
import os

class DiarizationResult:
    """Structured class for diarization results"""
    def __init__(
        self,
        segments: List[Dict[str, Any]],
        speakers: Dict[str, Dict[str, Any]],
        num_speakers: int,
        overlap_regions: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ):
        self.segments = segments
        self.speakers = speakers
        self.num_speakers = num_speakers
        self.overlap_regions = overlap_regions
        self.metadata = metadata
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'segments': self.segments,
            'speakers': self.speakers,
            'num_speakers': self.num_speakers,
            'overlap_regions': self.overlap_regions,
            'metadata': self.metadata
        }

class DiarizationException(Exception):
    """Custom exception for diarization errors"""
    pass

class Diarizer:
    """Enhanced speaker diarization with memory optimization and improved accuracy"""
    
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
        self.config = self._validate_config(config or {})
        
        # Initialize models
        self._initialize_models()
        
        # Setup processing pools with controlled parallelism
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 2)
        )
        
        # Progress tracking
        self.progress_callback = None
        self.diarization_progress = 0.0
        
        logger.info(f"Diarizer initialized on {self.device}")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default configuration."""
        defaults = {
            'min_speakers': 1,
            'max_speakers': 10,
            'threshold': 0.5,
            'overlap_threshold': 0.5,
            'min_segment_length': 0.5,  # seconds
            'max_workers': 2,           # control parallelism
            'chunk_size': 30,           # seconds
            'batch_size': 8,            # batch processing
            'max_memory_usage': 0.8,    # maximum memory fraction
            'sample_rate': 16000,
            'use_speechbrain': True,    # use speechbrain for embedding
            'use_resemblyzer': True,    # use resemblyzer for verification
            'cluster_method': 'agglomerative',  # clustering method
            'french_optimization': True,  # optimize for French speakers
            'segment_overlap': 0.2,     # overlap for segmentation
            'vad_filter': True,        # voice activity detection filtering
            'refinement_iterations': 2  # number of refinement iterations
        }
        
        # Update defaults with provided config
        validated_config = {**defaults, **(config or {})}
        
        # Validate values
        if validated_config['min_speakers'] < 1:
            validated_config['min_speakers'] = 1
        if validated_config['max_speakers'] < validated_config['min_speakers']:
            validated_config['max_speakers'] = validated_config['min_speakers']
        if validated_config['max_speakers'] > 20:
            validated_config['max_speakers'] = 20
            
        return validated_config

    def _initialize_models(self) -> None:
        """Initialize diarization models with proper memory management."""
        try:
            # Store initialization order to control loading
            self.initialization_state = {
                'pyannote': False,
                'ecapa': False,
                'resemblyzer': False
            }
            
            # Initialize PyAnnote pipeline
            self._init_pyannote()
            
            # Initialize ECAPA-TDNN for speaker embeddings
            self._init_ecapa()
            
            # Initialize Resemblyzer for verification
            self._init_resemblyzer()
            
            logger.info("Diarization models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing diarization models: {str(e)}")
            raise DiarizationException(f"Model initialization failed: {str(e)}")

    def _init_pyannote(self) -> None:
        """Initialize PyAnnote with disk caching."""
        try:
            # Create cache directory
            cache_dir = self.cache_manager.get_path("pyannote")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set environment variables to use cache
            os.environ['PYANNOTE_CACHE'] = str(cache_dir)
            
            # Initialize model with proper device
            self.pyannote = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.config.get('hf_token'),
                cache_dir=str(cache_dir)
            ).to(self.device)
            
            self.initialization_state['pyannote'] = True
            logger.info("PyAnnote initialized successfully")
            
        except Exception as e:
            logger.error(f"PyAnnote initialization failed: {str(e)}")
            raise

    def _init_ecapa(self) -> None:
        """Initialize ECAPA-TDNN with memory optimization."""
        try:
            if not self.config['use_speechbrain']:
                logger.info("Skipping ECAPA-TDNN initialization as per configuration")
                return
                
            self.ecapa_tdnn = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.cache_manager.get_path("ecapa_tdnn"),
                run_opts={"device": self.device}
            )
            
            self.initialization_state['ecapa'] = True
            logger.info("ECAPA-TDNN initialized successfully")
            
        except Exception as e:
            logger.error(f"ECAPA-TDNN initialization failed: {str(e)}")
            if self.initialization_state['pyannote']:
                logger.info("Will continue with PyAnnote only")

    def _init_resemblyzer(self) -> None:
        """Initialize Resemblyzer with memory optimization."""
        try:
            if not self.config['use_resemblyzer']:
                logger.info("Skipping Resemblyzer initialization as per configuration")
                return
                
            # Using CPU for Resemblyzer to save GPU memory
            device_type = "cpu"  # Always use CPU for this model
            
            self.resemblyzer = VoiceEncoder(
                device=device_type,
                weights_fpath=None  # Use default weights
            )
            
            self.initialization_state['resemblyzer'] = True
            logger.info("Resemblyzer initialized successfully")
            
        except Exception as e:
            logger.warning(f"Resemblyzer initialization failed: {str(e)}")
            logger.info("Will continue with available models")

    async def process(
        self,
        audio: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process audio for speaker diarization with enhanced accuracy and memory management.
        
        Args:
            audio: Preprocessed audio data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing diarization results
        """
        try:
            start_time = time.time()
            self.progress_callback = progress_callback
            self.diarization_progress = 0.0
            
            # Update progress
            await self._update_progress(0.0, "Starting speaker diarization")
            
            # Check prerequisites
            if not self.initialization_state['pyannote']:
                raise DiarizationException("PyAnnote model is required but not initialized")
            
            # Manage memory before processing
            await self._manage_memory()
            
            # Update progress
            await self._update_progress(0.1, "Initial diarization")
            
            # Initial diarization with PyAnnote
            pyannote_results = await self._pyannote_diarization(audio)
            
            # Update progress
            await self._update_progress(0.3, "Extracting speaker embeddings")
            
            # Extract speaker embeddings - conditional on available models
            if self.initialization_state['ecapa'] or self.initialization_state['resemblyzer']:
                embeddings = await self._extract_speaker_embeddings(
                    audio,
                    pyannote_results['segments']
                )
            else:
                embeddings = {}
            
            # Update progress
            await self._update_progress(0.5, "Refining speaker clusters")
            
            # Refine speaker clustering if embeddings are available
            if embeddings:
                refined_results = await self._refine_speaker_clusters(
                    pyannote_results['segments'],
                    embeddings
                )
            else:
                refined_results = pyannote_results
            
            # Update progress
            await self._update_progress(0.7, "Detecting speech overlap")
            
            # Detect overlapping speech
            overlap_regions = await self._detect_speech_overlap(
                audio,
                refined_results['segments']
            )
            
            # Update progress
            await self._update_progress(0.8, "Analyzing speaker characteristics")
            
            # Analyze speaker characteristics if French optimization is enabled
            if self.config['french_optimization']:
                await self._optimize_french_speakers(
                    audio,
                    refined_results['segments'],
                    embeddings
                )
            
            # Update progress
            await self._update_progress(0.9, "Finalizing results")
            
            # Combine and enhance results
            final_results = self._combine_results(
                refined_results,
                overlap_regions,
                time.time() - start_time
            )
            
            # Clean up memory
            await self._manage_memory()
            
            # Update progress
            await self._update_progress(1.0, "Diarization complete")
            
            # Return as structured result
            return DiarizationResult(
                segments=final_results['segments'],
                speakers=final_results['speakers'],
                num_speakers=final_results['num_speakers'],
                overlap_regions=final_results['overlap_regions'],
                metadata=final_results['metadata']
            ).to_dict()
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise DiarizationException(f"Diarization failed: {str(e)}")
        finally:
            # Ensure memory is cleaned up
            await self._manage_memory()

    async def _update_progress(self, progress: float, status: str) -> None:
        """Update diarization progress."""
        self.diarization_progress = progress
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
                
        except Exception as e:
            logger.warning(f"Memory management failed: {str(e)}")

    async def _pyannote_diarization(
        self,
        audio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform initial diarization using PyAnnote with optimizations."""
        loop = asyncio.get_event_loop()
        
        def _diarize():
            try:
                # Convert audio to PyAnnote format
                waveform = torch.tensor(audio['audio']).unsqueeze(0)
                sample_rate = audio.get('sample_rate', self.config['sample_rate'])
                
                # Run diarization with memory-optimized settings
                diarization = self.pyannote(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    min_speakers=self.config['min_speakers'],
                    max_speakers=self.config['max_speakers'],
                    num_workers=1  # Limit workers to save memory
                )
                
                # Extract segments with confidence estimation
                segments = []
                speaker_map = {}
                speaker_idx = 0
                
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Map original speaker ID to simplified format
                    if speaker not in speaker_map:
                        speaker_map[speaker] = f"speaker_{speaker_idx}"
                        speaker_idx += 1
                    
                    # Estimate confidence based on segment duration and position
                    # (longer segments typically have higher confidence)
                    segment_duration = turn.end - turn.start
                    confidence = min(max(0.5 + (segment_duration - 1.0) * 0.1, 0.5), 0.95)
                    
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker_id': speaker_map[speaker],
                        'confidence': confidence,
                        'duration': segment_duration
                    })
                
                return {
                    'segments': segments,
                    'num_speakers': len(speaker_map),
                    'speaker_map': speaker_map
                }
                
            except Exception as e:
                logger.error(f"PyAnnote diarization failed: {str(e)}")
                raise DiarizationException(f"PyAnnote diarization failed: {str(e)}")
        
        return await loop.run_in_executor(self.thread_pool, _diarize)

    async def _extract_speaker_embeddings(
        self,
        audio: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract speaker embeddings using available models with batched processing."""
        if not segments:
            return {}
            
        embeddings = {}
        sample_rate = audio.get('sample_rate', self.config['sample_rate'])
        audio_data = audio['audio']
        
        # Group segments by speaker for batch processing
        speaker_segments = {}
        for segment in segments:
            speaker_id = segment['speaker_id']
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)
        
        # Process each speaker's segments
        for speaker_id, spk_segments in speaker_segments.items():
            # Select up to 5 longest segments per speaker for embedding
            selected_segments = sorted(
                spk_segments, 
                key=lambda s: s['duration'], 
                reverse=True
            )[:5]
            
            speaker_embeddings = {'ecapa': None, 'resemblyzer': None}
            
            if self.initialization_state['ecapa']:
                ecapa_results = await asyncio.gather(*[
                    self._get_ecapa_embedding(
                        audio_data,
                        segment['start'],
                        segment['end'],
                        sample_rate
                    )
                    for segment in selected_segments
                ])
                
                # Average embeddings
                valid_embeddings = [e for e in ecapa_results if e is not None]
                if valid_embeddings:
                    speaker_embeddings['ecapa'] = torch.stack(valid_embeddings).mean(0)
            
            if self.initialization_state['resemblyzer']:
                resem_results = await asyncio.gather(*[
                    self._get_resemblyzer_embedding(
                        audio_data,
                        segment['start'],
                        segment['end'],
                        sample_rate
                    )
                    for segment in selected_segments
                ])
                
                # Average embeddings
                valid_embeddings = [e for e in resem_results if e is not None]
                if valid_embeddings:
                    speaker_embeddings['resemblyzer'] = torch.stack(valid_embeddings).mean(0)
            
            # Only add speaker if at least one embedding method worked
            if speaker_embeddings['ecapa'] is not None or speaker_embeddings['resemblyzer'] is not None:
                embeddings[speaker_id] = {
                    k: v for k, v in speaker_embeddings.items() if v is not None
                }
        
        return embeddings

    async def _get_ecapa_embedding(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int
    ) -> Optional[torch.Tensor]:
        """Get ECAPA-TDNN embedding with error handling."""
        if not self.initialization_state['ecapa']:
            return None
            
        loop = asyncio.get_event_loop()
        
        def _extract():
            try:
                # Convert time to samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Ensure within bounds
                if start_sample >= len(audio) or end_sample <= start_sample:
                    return None
                    
                end_sample = min(end_sample, len(audio))
                
                # Extract segment
                segment_audio = audio[start_sample:end_sample]
                
                # Need minimum length for processing
                if len(segment_audio) < 0.5 * sample_rate:
                    return None
                
                # Convert to torch tensor
                segment_tensor = torch.tensor(segment_audio).float().to(self.device)
                
                with torch.no_grad():
                    return self.ecapa_tdnn.encode_batch(segment_tensor).squeeze()
            except Exception as e:
                logger.warning(f"ECAPA embedding extraction failed: {str(e)}")
                return None
        
        return await loop.run_in_executor(self.thread_pool, _extract)

    async def _get_resemblyzer_embedding(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        sample_rate: int
    ) -> Optional[torch.Tensor]:
        """Get Resemblyzer embedding with error handling."""
        if not self.initialization_state['resemblyzer']:
            return None
            
        loop = asyncio.get_event_loop()
        
        def _extract():
            try:
                # Convert time to samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Ensure within bounds
                if start_sample >= len(audio) or end_sample <= start_sample:
                    return None
                    
                end_sample = min(end_sample, len(audio))
                
                # Extract segment
                segment_audio = audio[start_sample:end_sample]
                
                # Need minimum length for processing
                if len(segment_audio) < 1.0 * sample_rate:
                    return None
                
                # Process with resemblyzer
                processed = preprocess_wav(segment_audio, sample_rate)
                embedding = self.resemblyzer.embed_utterance(processed)
                
                return torch.from_numpy(embedding).to(self.device)
            except Exception as e:
                logger.warning(f"Resemblyzer embedding extraction failed: {str(e)}")
                return None
        
        return await loop.run_in_executor(self.thread_pool, _extract)

    async def _refine_speaker_clusters(
        self,
        segments: List[Dict[str, Any]],
        embeddings: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Refine speaker clusters using embeddings."""
        # Skip refinement if not enough speakers or embeddings
        if len(embeddings) <= 1:
            return {'segments': segments, 'num_speakers': len(embeddings)}
            
        try:
            # Prepare speaker list
            speakers = list(embeddings.keys())
            n_speakers = len(speakers)
            
            # Create distance matrix
            # Note: We use a similarity matrix and convert to distance
            similarity_matrix = torch.zeros((n_speakers, n_speakers))
            
            # Prioritize embeddings: both > ecapa > resemblyzer
            for i, spk1 in enumerate(speakers):
                for j, spk2 in enumerate(speakers):
                    if i < j:  # Only compute upper triangle
                        # Determine which embeddings to use
                        if 'ecapa' in embeddings[spk1] and 'ecapa' in embeddings[spk2]:
                            # ECAPA embeddings
                            ecapa_sim = F.cosine_similarity(
                                embeddings[spk1]['ecapa'].unsqueeze(0),
                                embeddings[spk2]['ecapa'].unsqueeze(0)
                            )
                            
                            if 'resemblyzer' in embeddings[spk1] and 'resemblyzer' in embeddings[spk2]:
                                # Both available - weighted combination
                                resem_sim = F.cosine_similarity(
                                    embeddings[spk1]['resemblyzer'].unsqueeze(0),
                                    embeddings[spk2]['resemblyzer'].unsqueeze(0)
                                )
                                sim = 0.7 * ecapa_sim + 0.3 * resem_sim
                            else:
                                # Only ECAPA
                                sim = ecapa_sim
                        elif 'resemblyzer' in embeddings[spk1] and 'resemblyzer' in embeddings[spk2]:
                            # Only Resemblyzer
                            sim = F.cosine_similarity(
                                embeddings[spk1]['resemblyzer'].unsqueeze(0),
                                embeddings[spk2]['resemblyzer'].unsqueeze(0)
                            )
                        else:
                            # No matching embeddings - use default
                            sim = torch.tensor(0.0)
                        
                        # Convert similarity to distance (1 - similarity)
                        # and fill both i,j and j,i positions
                        dist = 1.0 - sim
                        similarity_matrix[i, j] = similarity_matrix[j, i] = dist
            
            # Create distance matrix for clustering
            distance_matrix = similarity_matrix.cpu().numpy()
            
            # Perform clustering based on selected method
            if self.config['cluster_method'] == 'agglomerative':
                # Agglomerative clustering with threshold
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.config['threshold'],
                    affinity='precomputed',
                    linkage='average'  # Use average linkage for better stability
                )
                
                labels = clustering.fit_predict(distance_matrix)
                
                # Limit number of speakers if necessary
                unique_labels = np.unique(labels)
                if len(unique_labels) > self.config['max_speakers']:
                    # Re-cluster with fixed number of clusters
                    clustering = AgglomerativeClustering(
                        n_clusters=self.config['max_speakers'],
                        affinity='precomputed',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                    unique_labels = np.unique(labels)
            elif self.config['cluster_method'] == 'spectral':
                # Spectral clustering for more complex patterns
                from sklearn.cluster import SpectralClustering
                
                # Use similarity instead of distance for spectral clustering
                affinity_matrix = 1.0 - distance_matrix
                
                # Apply spectral clustering
                clustering = SpectralClustering(
                    n_clusters=min(n_speakers, self.config['max_speakers']),
                    affinity='precomputed',
                    random_state=42,
                    n_init=5
                )
                
                labels = clustering.fit_predict(affinity_matrix)
                unique_labels = np.unique(labels)
            else:
                # Default to agglomerative
                clustering = AgglomerativeClustering(
                    n_clusters=min(n_speakers, self.config['max_speakers']),
                    affinity='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
                unique_labels = np.unique(labels)
            
            # Create speaker mapping
            speaker_map = {speakers[i]: f"speaker_{labels[i]}" for i in range(n_speakers)}
            
            # Update segment speaker IDs and compute confidences
            new_segments = []
            for segment in segments:
                old_speaker_id = segment['speaker_id']
                
                # Use mapping if available, otherwise keep original
                new_speaker_id = speaker_map.get(old_speaker_id, old_speaker_id)
                
                # Compute confidence based on original confidence and embedding match
                confidence = segment['confidence']
                
                # Add refined segment
                new_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker_id': new_speaker_id,
                    'confidence': confidence,
                    'duration': segment['duration']
                })
            
            return {
                'segments': new_segments,
                'num_speakers': len(unique_labels)
            }
            
        except Exception as e:
            logger.warning(f"Speaker refinement failed: {str(e)}, using original segmentation")
            return {'segments': segments, 'num_speakers': len(set(s['speaker_id'] for s in segments))}

    async def _detect_speech_overlap(
        self,
        audio: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect regions of overlapping speech with enhanced accuracy."""
        loop = asyncio.get_event_loop()
        
        def _detect():
            try:
                overlap_regions = []
                
                # Sort segments by start time
                sorted_segments = sorted(segments, key=lambda x: x['start'])
                
                # Check for temporal overlaps
                for i in range(len(sorted_segments) - 1):
                    current = sorted_segments[i]
                    j = i + 1
                    
                    # Check all potentially overlapping segments
                    while j < len(sorted_segments) and sorted_segments[j]['start'] < current['end']:
                        next_seg = sorted_segments[j]
                        
                        # Calculate overlap
                        overlap_start = max(current['start'], next_seg['start'])
                        overlap_end = min(current['end'], next_seg['end'])
                        
                        # Only consider significant overlaps
                        if (overlap_end > overlap_start and 
                            current['speaker_id'] != next_seg['speaker_id'] and
                            overlap_end - overlap_start >= self.config['min_segment_length']):
                            
                            # Calculate overlap ratio
                            overlap_duration = overlap_end - overlap_start
                            min_segment_duration = min(
                                current['end'] - current['start'],
                                next_seg['end'] - next_seg['start']
                            )
                            overlap_ratio = overlap_duration / min_segment_duration
                            
                            # Only add if overlap is significant
                            if overlap_ratio >= self.config['overlap_threshold']:
                                overlap_regions.append({
                                    'start': overlap_start,
                                    'end': overlap_end,
                                    'speakers': [current['speaker_id'], next_seg['speaker_id']],
                                    'confidence': min(current['confidence'], next_seg['confidence']),
                                    'duration': overlap_duration
                                })
                        
                        j += 1
                
                # Merge adjacent overlaps with same speakers
                if overlap_regions:
                    merged_overlaps = [overlap_regions[0]]
                    for current in overlap_regions[1:]:
                        prev = merged_overlaps[-1]
                        
                        # If same speakers and close in time, merge
                        if (set(current['speakers']) == set(prev['speakers']) and 
                            current['start'] - prev['end'] < 0.5):
                            
                            # Extend previous region
                            prev['end'] = current['end']
                            prev['duration'] = prev['end'] - prev['start']
                            prev['confidence'] = (prev['confidence'] + current['confidence']) / 2
                        else:
                            # Add as new region
                            merged_overlaps.append(current)
                    
                    return merged_overlaps
                
                return []
                
            except Exception as e:
                logger.warning(f"Overlap detection failed: {str(e)}")
                return []
        
        return await loop.run_in_executor(self.thread_pool, _detect)

    async def _optimize_french_speakers(
        self,
        audio: Dict[str, Any],
        segments: List[Dict[str, Any]],
        embeddings: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        """Optimize speaker identification for French content."""
        if not self.config['french_optimization'] or not segments:
            return
            
        try:
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker_id = segment['speaker_id']
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(segment)
            
            # Apply optimizations for each speaker
            for speaker_id, speaker_segs in speaker_segments.items():
                # Calculate average speaking rate
                total_words = sum(len(seg.get('text', '').split()) for seg in speaker_segs if 'text' in seg)
                total_duration = sum(seg['duration'] for seg in speaker_segs)
                
                if total_words > 0 and total_duration > 0:
                    speaking_rate = total_words / total_duration
                    
                    # Update confidence based on speaking rate patterns typical in French
                    # (French speakers often have consistent speaking rates)
                    if 2.0 <= speaking_rate <= 4.0:  # Typical French speaking rate
                        for seg in speaker_segs:
                            seg['confidence'] = min(seg['confidence'] * 1.1, 0.99)
                
                # Check for turn-taking patterns typical in French conversations
                if len(speaker_segments) > 1:
                    self._analyze_turn_taking(speaker_segments)
        except Exception as e:
            logger.warning(f"French speaker optimization failed: {str(e)}")

    def _analyze_turn_taking(self, speaker_segments: Dict[str, List[Dict[str, Any]]]) -> None:
        """Analyze turn-taking patterns to improve speaker confidence scores."""
        # Flatten segments for timeline analysis
        all_segments = []
        for speaker_id, segments in speaker_segments.items():
            for segment in segments:
                all_segments.append({
                    **segment,
                    'speaker_id': speaker_id
                })
        
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])
        
        # Analyze turn transitions
        for i in range(len(all_segments) - 1):
            current = all_segments[i]
            next_seg = all_segments[i + 1]
            
            # Only analyze transitions between different speakers
            if current['speaker_id'] != next_seg['speaker_id']:
                # Calculate gap between segments
                gap = next_seg['start'] - current['end']
                
                # Typical turn transitions in conversation have small gaps
                if 0 <= gap <= 1.0:  # Natural turn-taking gap (0-1 seconds)
                    # Boost confidence for both segments
                    current['confidence'] = min(current['confidence'] * 1.05, 0.99)
                    next_seg['confidence'] = min(next_seg['confidence'] * 1.05, 0.99)

    def _combine_results(
        self,
        diarization_results: Dict[str, Any],
        overlap_regions: List[Dict[str, Any]],
        processing_time: float
    ) -> Dict[str, Any]:
        """Combine and format diarization results."""
        # Get unique speakers and their segments
        speakers = {}
        segments = diarization_results['segments']
        
        for segment in segments:
            speaker_id = segment['speaker_id']
            if speaker_id not in speakers:
                speakers[speaker_id] = {
                    'total_duration': 0.0,
                    'segments': [],
                    'avg_confidence': 0.0,
                    'overlap_duration': 0.0,
                    'speaking_rate': 0.0,
                    'segment_count': 0,
                    'avg_segment_duration': 0.0
                }
            
            duration = segment['end'] - segment['start']
            speakers[speaker_id]['total_duration'] += duration
            speakers[speaker_id]['segments'].append(segment)
            speakers[speaker_id]['avg_confidence'] += (segment['confidence'] * duration)
            speakers[speaker_id]['segment_count'] += 1
        
        # Calculate overlap durations per speaker
        for overlap in overlap_regions:
            for speaker_id in overlap['speakers']:
                if speaker_id in speakers:
                    speakers[speaker_id]['overlap_duration'] += overlap['duration']
        
        # Compute speaker statistics
        for speaker_id, info in speakers.items():
            total_duration = info['total_duration']
            segment_count = info['segment_count']
            
            if total_duration > 0:
                info['avg_confidence'] /= total_duration
                info['avg_segment_duration'] = total_duration / segment_count if segment_count > 0 else 0
                
                # Calculate speaking rate if text is available
                word_count = sum(len(seg.get('text', '').split()) for seg in info['segments'] if 'text' in seg)
                if word_count > 0:
                    info['speaking_rate'] = word_count / total_duration
        
        # Format final speaker info
        speaker_info = {
            speaker_id: {
                'total_duration': info['total_duration'],
                'avg_confidence': info['avg_confidence'],
                'segment_count': info['segment_count'],
                'avg_segment_duration': info['avg_segment_duration'],
                'overlap_duration': info['overlap_duration'],
                'overlap_percentage': (info['overlap_duration'] / info['total_duration'] * 100) 
                                       if info['total_duration'] > 0 else 0,
                'speaking_rate': info['speaking_rate']
            }
            for speaker_id, info in speakers.items()
        }
        
        # Final results
        return {
            'segments': segments,
            'overlap_regions': overlap_regions,
            'speakers': speaker_info,
            'num_speakers': diarization_results['num_speakers'],
            'metadata': {
                'processing_time': processing_time,
                'overlap_count': len(overlap_regions),
                'total_overlap_duration': sum(o['duration'] for o in overlap_regions),
                'total_speech_duration': sum(info['total_duration'] for info in speaker_info.values()),
                'speaker_balancing': self._calculate_speaker_balance(speaker_info)
            }
        }

    def _calculate_speaker_balance(self, speaker_info: Dict[str, Dict[str, Any]]) -> float:
        """Calculate speaker balance ratio (0-1, higher is more balanced)."""
        if not speaker_info:
            return 0.0
            
        durations = [info['total_duration'] for info in speaker_info.values()]
        
        if not durations or sum(durations) == 0:
            return 0.0
            
        # Calculate Gini coefficient (measure of inequality)
        n = len(durations)
        if n <= 1:
            return 1.0  # Perfect balance with single speaker
            
        # Normalize durations
        total_duration = sum(durations)
        proportions = [d / total_duration for d in durations]
        
        # Sort proportions
        sorted_proportions = sorted(proportions)
        
        # Calculate Gini coefficient
        cumulative = 0
        for i, proportion in enumerate(sorted_proportions):
            cumulative += proportion * (n - i - 0.5)
        
        gini = 2 * cumulative / n - 1
        
        # Convert Gini to balance score (1 - Gini)
        # Gini of 0 means perfect equality, so balance is 1
        return 1.0 - gini

    async def analyze_speaker_characteristics(
        self,
        audio: Dict[str, Any],
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze speaker characteristics (pitch, energy, etc.)."""
        try:
            if not segments:
                return {}
                
            sample_rate = audio.get('sample_rate', self.config['sample_rate'])
            audio_data = audio['audio']
            
            # Group segments by speaker
            speaker_segments = {}
            for segment in segments:
                speaker_id = segment['speaker_id']
                if speaker_id not in speaker_segments:
                    speaker_segments[speaker_id] = []
                speaker_segments[speaker_id].append(segment)
            
            # Analyze each speaker
            speaker_characteristics = {}
            
            for speaker_id, speaker_segs in speaker_segments.items():
                # Get longest segments for analysis (up to 3)
                analysis_segments = sorted(
                    speaker_segs,
                    key=lambda s: s['duration'],
                    reverse=True
                )[:3]
                
                # Skip if no valid segments
                if not analysis_segments:
                    continue
                
                # Extract audio for each segment
                segment_features = []
                
                for segment in analysis_segments:
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    
                    # Skip invalid boundaries
                    if start_sample >= len(audio_data) or end_sample <= start_sample:
                        continue
                        
                    # Limit to audio length
                    end_sample = min(end_sample, len(audio_data))
                    
                    # Extract segment audio
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    # Skip segments that are too short
                    if len(segment_audio) < 0.5 * sample_rate:
                        continue
                    
                    # Compute basic acoustic features
                    features = await self._compute_acoustic_features(
                        segment_audio,
                        sample_rate
                    )
                    
                    if features:
                        segment_features.append(features)
                
                # Average features across segments
                if segment_features:
                    avg_features = {}
                    for key in segment_features[0].keys():
                        if key != 'pitch_contour' and key != 'energy_contour':
                            avg_features[key] = sum(f[key] for f in segment_features) / len(segment_features)
                    
                    speaker_characteristics[speaker_id] = avg_features
            
            return speaker_characteristics
            
        except Exception as e:
            logger.warning(f"Speaker characteristic analysis failed: {str(e)}")
            return {}

    async def _compute_acoustic_features(
        self,
        audio_segment: np.ndarray,
        sample_rate: int
    ) -> Optional[Dict[str, Any]]:
        """Compute acoustic features for a segment."""
        try:
            import librosa
            
            # Compute pitch (F0) using PYIN
            pitches, voiced_flags = librosa.piptrack(
                y=audio_segment, 
                sr=sample_rate,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            # Extract mean pitch from voiced frames
            voiced_pitches = []
            for i in range(pitches.shape[1]):
                voiced_indices = np.where(voiced_flags[:, i])[0]
                if voiced_indices.size > 0:
                    voiced_pitches.extend(pitches[voiced_indices, i])
            
            # Calculate pitch statistics
            if voiced_pitches:
                mean_pitch = np.mean(voiced_pitches)
                std_pitch = np.std(voiced_pitches)
            else:
                mean_pitch = 0.0
                std_pitch = 0.0
            
            # Compute energy
            energy = np.mean(audio_segment**2)
            
            # Compute spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio_segment, sr=sample_rate
            )[0])
            
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=audio_segment, sr=sample_rate
            )[0])
            
            # Return features
            return {
                'mean_pitch': float(mean_pitch),
                'pitch_std': float(std_pitch),
                'energy': float(energy),
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth)
            }
            
        except Exception as e:
            logger.warning(f"Acoustic feature computation failed: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clear model references
            self.pyannote = None
            self.ecapa_tdnn = None
            self.resemblyzer = None
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Diarizer resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")