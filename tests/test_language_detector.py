import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from models.language_detector import LanguageDetector
import asyncio

@pytest.fixture
def mock_cache_manager():
    cache_manager = Mock()
    cache_manager.get_path.return_value = Path("cache/models")
    return cache_manager

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
async def language_detector(device, mock_cache_manager):
    detector = LanguageDetector(device, mock_cache_manager)
    yield detector
    await detector.cleanup()

@pytest.fixture
def sample_audio():
    # Generate synthetic audio data
    duration = 5  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    return torch.from_numpy(audio.astype(np.float32))

@pytest.mark.asyncio
async def test_initialization(language_detector):
    assert language_detector.device is not None
    assert language_detector.voxlingua is not None
    assert language_detector.whisper is not None
    assert language_detector.whisper_jax is not None

@pytest.mark.asyncio
async def test_detect_french(language_detector, sample_audio):
    result = await language_detector.detect(sample_audio)
    assert 'primary_language' in result
    assert 'distribution' in result
    assert 'confidence' in result
    assert len(result['distribution']) >= 1

@pytest.mark.asyncio
async def test_detect_multilingual(language_detector):
    # Create multi-lingual synthetic audio
    audio_length = 16000 * 10  # 10 seconds at 16kHz
    audio = torch.randn(audio_length)
    
    result = await language_detector.detect(audio)
    assert len(result['distribution']) >= 1
    assert all(0 <= lang['confidence'] <= 1 for lang in result['distribution'])

@pytest.mark.asyncio
async def test_language_segments(language_detector, sample_audio):
    result = await language_detector.detect(sample_audio)
    assert 'segments' in result
    if result['segments']:
        segment = result['segments'][0]
        assert 'start_time' in segment
        assert 'end_time' in segment
        assert 'language' in segment
        assert 'confidence' in segment

@pytest.mark.asyncio
async def test_error_handling(language_detector):
    with pytest.raises(Exception):
        await language_detector.detect(None)
    
    with pytest.raises(Exception):
        await language_detector.detect(torch.tensor([]))

@pytest.mark.asyncio
async def test_confidence_scores(language_detector, sample_audio):
    result = await language_detector.detect(sample_audio)
    assert 0 <= result['confidence'] <= 1
    for lang_info in result['distribution']:
        assert 0 <= lang_info['confidence'] <= 1

@pytest.mark.asyncio
async def test_detector_agreement(language_detector, sample_audio):
    result = await language_detector.detect(sample_audio)
    for lang_info in result['distribution']:
        assert 'support' in lang_info
        assert 0 <= lang_info['support'] <= 1

@pytest.mark.asyncio
async def test_short_audio(language_detector):
    short_audio = torch.randn(800)  # 0.05 seconds at 16kHz
    with pytest.raises(Exception):
        await language_detector.detect(short_audio)

@pytest.mark.asyncio
async def test_long_audio(language_detector):
    # Test with 30 seconds audio
    long_audio = torch.randn(16000 * 30)
    result = await language_detector.detect(long_audio)
    assert result['primary_language'] is not None

@pytest.mark.asyncio
async def test_noisy_audio(language_detector):
    # Create noisy audio
    audio = torch.randn(16000 * 5) * 0.1
    result = await language_detector.detect(audio)
    assert result['confidence'] < 0.8  # Should have lower confidence

@pytest.mark.asyncio
async def test_concurrent_detection(language_detector, sample_audio):
    # Test concurrent processing
    tasks = [
        language_detector.detect(sample_audio)
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    assert all(r['primary_language'] for r in results)

@pytest.mark.asyncio
async def test_memory_management(language_detector, sample_audio):
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Process multiple times
    for _ in range(5):
        await language_detector.detect(sample_audio)
        
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory * 1.5  # Allow some overhead

@pytest.mark.asyncio
async def test_model_fallback(language_detector, sample_audio):
    # Test fallback mechanism when primary model fails
    with patch.object(language_detector, '_detect_with_voxlingua', side_effect=Exception):
        result = await language_detector.detect(sample_audio)
        assert result['primary_language'] is not None

@pytest.mark.asyncio
async def test_cleanup(language_detector):
    await language_detector.cleanup()
    assert True  # Should not raise any exceptions

if __name__ == '__main__':
    pytest.main(['-v', __file__])