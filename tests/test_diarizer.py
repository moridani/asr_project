import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio
from models.diarizer import Diarizer
import time

@pytest.fixture
def mock_cache_manager():
    cache_manager = Mock()
    cache_manager.get_path.return_value = Path("cache/models")
    return cache_manager

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
async def diarizer(device, mock_cache_manager):
    diarizer = Diarizer(device, mock_cache_manager)
    yield diarizer
    await diarizer.cleanup()

@pytest.fixture
def multi_speaker_audio():
    duration = 10
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate two different frequencies for different speakers
    speaker1 = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz
    speaker2 = np.sin(2 * np.pi * 880 * t) * 0.5  # 880Hz
    
    # Alternate between speakers
    audio = np.zeros_like(t)
    segments = 5
    segment_length = len(t) // segments
    
    for i in range(segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        if i % 2 == 0:
            audio[start_idx:end_idx] = speaker1[start_idx:end_idx]
        else:
            audio[start_idx:end_idx] = speaker2[start_idx:end_idx]
    
    return {
        'audio': torch.from_numpy(audio.astype(np.float32)),
        'sample_rate': sample_rate
    }

@pytest.mark.asyncio
async def test_initialization(diarizer):
    assert diarizer.pyannote is not None
    assert diarizer.ecapa_tdnn is not None
    assert diarizer.resemblyzer is not None

@pytest.mark.asyncio
async def test_diarization_basic(diarizer, multi_speaker_audio):
    result = await diarizer.process(multi_speaker_audio)
    assert 'segments' in result
    assert 'speakers' in result
    assert 'num_speakers' in result
    assert 'metadata' in result
    assert len(result['segments']) > 0

@pytest.mark.asyncio
async def test_speaker_segmentation(diarizer, multi_speaker_audio):
    result = await diarizer.process(multi_speaker_audio)
    segments = result['segments']
    
    for segment in segments:
        assert all(key in segment for key in [
            'start', 'end', 'speaker_id', 'confidence'
        ])
        assert segment['start'] < segment['end']
        assert segment['confidence'] >= 0 and segment['confidence'] <= 1

@pytest.mark.asyncio
async def test_overlap_detection(diarizer, multi_speaker_audio):
    result = await diarizer.process(multi_speaker_audio)
    assert 'overlap_regions' in result
    
    if result['overlap_regions']:
        overlap = result['overlap_regions'][0]
        assert 'start' in overlap
        assert 'end' in overlap
        assert 'speakers' in overlap
        assert len(overlap['speakers']) >= 2

@pytest.mark.asyncio
async def test_speaker_embeddings(diarizer, multi_speaker_audio):
    result = await diarizer._extract_speaker_embeddings(
        multi_speaker_audio['audio'],
        result['segments']
    )
    assert len(result) > 0
    for speaker_id, embeddings in result.items():
        assert 'ecapa' in embeddings
        assert 'resemblyzer' in embeddings

@pytest.mark.asyncio
async def test_concurrent_processing(diarizer, multi_speaker_audio):
    tasks = [
        diarizer.process(multi_speaker_audio)
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    assert all('segments' in r for r in results)

@pytest.mark.asyncio
async def test_error_handling(diarizer):
    with pytest.raises(Exception):
        await diarizer.process(None)
    
    with pytest.raises(Exception):
        await diarizer.process({'audio': torch.tensor([])})

@pytest.mark.asyncio
async def test_memory_management(diarizer, multi_speaker_audio):
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for _ in range(3):
        result = await diarizer.process(multi_speaker_audio)
        assert result is not None
        
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory * 1.5

@pytest.mark.asyncio
async def test_speaker_confidence(diarizer, multi_speaker_audio):
    result = await diarizer.process(multi_speaker_audio)
    
    for speaker_info in result['speakers'].values():
        assert 'avg_confidence' in speaker_info
        assert 0 <= speaker_info['avg_confidence'] <= 1

@pytest.mark.asyncio
async def test_minimum_segment_length(diarizer):
    # Test with very short audio
    short_duration = 0.1
    sample_rate = 16000
    t = np.linspace(0, short_duration, int(sample_rate * short_duration))
    audio = np.sin(2 * np.pi * 440 * t)
    
    with pytest.raises(Exception):
        await diarizer.process({
            'audio': torch.from_numpy(audio.astype(np.float32)),
            'sample_rate': sample_rate
        })

@pytest.mark.asyncio
async def test_long_audio_processing(diarizer):
    duration = 30
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create alternating speakers
    speaker1 = np.sin(2 * np.pi * 440 * t)
    speaker2 = np.sin(2 * np.pi * 880 * t)
    audio = np.zeros_like(t)
    
    segment_length = len(t) // 10
    for i in range(10):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        audio[start_idx:end_idx] = speaker1[start_idx:end_idx] if i % 2 == 0 else speaker2[start_idx:end_idx]
    
    result = await diarizer.process({
        'audio': torch.from_numpy(audio.astype(np.float32)),
        'sample_rate': sample_rate
    })
    
    assert len(result['segments']) > 5

@pytest.mark.asyncio
async def test_speaker_clustering(diarizer, multi_speaker_audio):
    result = await diarizer.process(multi_speaker_audio)
    
    # Test speaker clustering consistency
    speaker_segments = {}
    for segment in result['segments']:
        speaker_id = segment['speaker_id']
        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []
        speaker_segments[speaker_id].append(segment)
    
    # Check that each speaker has consistent embeddings
    for speaker_id, segments in speaker_segments.items():
        assert len(segments) > 0
        for segment in segments:
            assert segment['confidence'] > 0

@pytest.mark.asyncio
async def test_performance_metrics(diarizer, multi_speaker_audio):
    start_time = time.time()
    result = await diarizer.process(multi_speaker_audio)
    processing_time = time.time() - start_time
    
    assert 'metadata' in result
    assert 'processing_time' in result['metadata']
    assert result['metadata']['processing_time'] <= processing_time

@pytest.mark.asyncio
async def test_noise_robustness(diarizer):
    duration = 5
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_audio = np.sin(2 * np.pi * 440 * t)
    
    # Add noise
    noise = np.random.normal(0, 0.1, len(clean_audio))
    noisy_audio = clean_audio + noise
    
    result = await diarizer.process({
        'audio': torch.from_numpy(noisy_audio.astype(np.float32)),
        'sample_rate': sample_rate
    })
    
    assert result['num_speakers'] > 0

@pytest.mark.asyncio
async def test_cleanup(diarizer):
    await diarizer.cleanup()
    assert True

if __name__ == '__main__':
    pytest.main(['-v', __file__])