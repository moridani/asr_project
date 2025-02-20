import pytest
import torch
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio
from models.transcriber import Transcriber

@pytest.fixture
def mock_cache_manager():
    cache_manager = Mock()
    cache_manager.get_path.return_value = Path("cache/models")
    return cache_manager

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
async def transcriber(device, mock_cache_manager):
    transcriber = Transcriber(device, mock_cache_manager)
    yield transcriber
    await transcriber.cleanup()

@pytest.fixture
def audio_data():
    duration = 5
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    return {
        'audio': torch.from_numpy(audio.astype(np.float32)),
        'sample_rate': sample_rate
    }

@pytest.mark.asyncio
async def test_initialization(transcriber):
    assert transcriber.whisper is not None
    assert transcriber.conformer is not None
    assert transcriber.custom_vocab is not None

@pytest.mark.asyncio
async def test_transcription_basic(transcriber, audio_data):
    result = await transcriber.transcribe(audio_data, 'en')
    assert 'segments' in result
    assert 'text' in result
    assert 'confidence' in result
    assert 'language' in result
    assert result['language'] == 'en'

@pytest.mark.asyncio
async def test_transcription_segments(transcriber, audio_data):
    result = await transcriber.transcribe(audio_data, 'en')
    assert len(result['segments']) > 0
    segment = result['segments'][0]
    assert all(key in segment for key in [
        'text', 'start_time', 'end_time', 'confidence'
    ])

@pytest.mark.asyncio
async def test_custom_vocabulary(transcriber, audio_data):
    # Test with custom vocabulary
    test_vocab = {'en': {'testtermx': 'test term x'}}
    with patch.object(transcriber, 'custom_vocab', test_vocab):
        result = await transcriber.transcribe(audio_data, 'en')
        assert result is not None

@pytest.mark.asyncio
async def test_different_languages(transcriber, audio_data):
    languages = ['en', 'fr', 'es', 'de']
    results = []
    for lang in languages:
        result = await transcriber.transcribe(audio_data, lang)
        results.append(result)
        assert result['language'] == lang

@pytest.mark.asyncio
async def test_model_fallback(transcriber, audio_data):
    # Force Whisper to fail to test Conformer fallback
    with patch.object(transcriber, '_transcribe_with_whisper', side_effect=Exception):
        result = await transcriber.transcribe(audio_data, 'en')
        assert result['metadata']['model_used'] == 'conformer'

@pytest.mark.asyncio
async def test_error_handling(transcriber):
    with pytest.raises(Exception):
        await transcriber.transcribe(None, 'en')
    
    with pytest.raises(Exception):
        await transcriber.transcribe({'audio': torch.tensor([])}, 'en')

@pytest.mark.asyncio
async def test_concurrent_transcription(transcriber, audio_data):
    tasks = [
        transcriber.transcribe(audio_data, 'en')
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    assert all('text' in r for r in results)

@pytest.mark.asyncio
async def test_long_audio(transcriber):
    # Test with 30 seconds audio
    duration = 30
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    audio_data = {
        'audio': torch.from_numpy(audio.astype(np.float32)),
        'sample_rate': sample_rate
    }
    
    result = await transcriber.transcribe(audio_data, 'en')
    assert len(result['segments']) > 1

@pytest.mark.asyncio
async def test_memory_management(transcriber, audio_data):
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for _ in range(5):
        result = await transcriber.transcribe(audio_data, 'en')
        assert result is not None
        
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory * 1.5

@pytest.mark.asyncio
async def test_noisy_audio(transcriber):
    duration = 5
    sample_rate = 16000
    noise = np.random.normal(0, 0.1, int(sample_rate * duration))
    audio_data = {
        'audio': torch.from_numpy(noise.astype(np.float32)),
        'sample_rate': sample_rate
    }
    
    result = await transcriber.transcribe(audio_data, 'en')
    assert result['confidence'] < 0.8

@pytest.mark.asyncio
async def test_confidence_thresholds(transcriber, audio_data):
    result = await transcriber.transcribe(audio_data, 'en')
    assert all(0 <= seg['confidence'] <= 1 for seg in result['segments'])

@pytest.mark.asyncio
async def test_timestamp_consistency(transcriber, audio_data):
    result = await transcriber.transcribe(audio_data, 'en')
    segments = result['segments']
    
    # Check timestamp ordering
    for i in range(len(segments) - 1):
        assert segments[i]['end_time'] <= segments[i + 1]['start_time']
        assert segments[i]['start_time'] < segments[i]['end_time']

@pytest.mark.asyncio
async def test_empty_segments_handling(transcriber, audio_data):
    # Test handling of silence/empty segments
    silence = np.zeros(16000 * 2)  # 2 seconds of silence
    audio_data['audio'] = torch.from_numpy(silence.astype(np.float32))
    
    result = await transcriber.transcribe(audio_data, 'en')
    assert result is not None
    assert isinstance(result['segments'], list)

@pytest.mark.asyncio
async def test_cleanup(transcriber):
    await transcriber.cleanup()
    assert True  # Should not raise any exceptions

@pytest.mark.asyncio
async def test_performance_metrics(transcriber, audio_data):
    start_time = time.time()
    result = await transcriber.transcribe(audio_data, 'en')
    processing_time = time.time() - start_time
    
    assert 'metadata' in result
    assert 'processing_time' in result['metadata']
    assert result['metadata']['processing_time'] <= processing_time

@pytest.mark.asyncio
async def test_batch_processing(transcriber):
    batch_size = 3
    duration = 2
    sample_rate = 16000
    
    # Create batch of audio samples
    audio_batch = []
    for _ in range(batch_size):
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        audio_batch.append({
            'audio': torch.from_numpy(audio.astype(np.float32)),
            'sample_rate': sample_rate
        })
    
    tasks = [transcriber.transcribe(audio, 'en') for audio in audio_batch]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == batch_size
    assert all('text' in r for r in results)

if __name__ == '__main__':
    pytest.main(['-v', __file__])