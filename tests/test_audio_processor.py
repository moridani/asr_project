import pytest
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
from unittest.mock import Mock, patch
import asyncio
from utils.audio_processor import AudioProcessor
import tempfile
import os

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
async def audio_processor(device):
    processor = AudioProcessor(device)
    yield processor
    await processor.cleanup()

@pytest.fixture
def sample_audio_file():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        sf.write(temp_file.name, audio, sample_rate)
        yield temp_file.name
    os.unlink(temp_file.name)

@pytest.fixture
def noisy_audio_file():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        clean_audio = np.sin(2 * np.pi * 440 * t)
        noise = np.random.normal(0, 0.1, len(clean_audio))
        noisy_audio = clean_audio + noise
        sf.write(temp_file.name, noisy_audio, sample_rate)
        yield temp_file.name
    os.unlink(temp_file.name)

@pytest.mark.asyncio
async def test_load_and_preprocess(audio_processor, sample_audio_file):
    result = await audio_processor.load_and_preprocess(sample_audio_file)
    assert 'audio' in result
    assert 'sample_rate' in result
    assert 'duration' in result
    assert 'quality_metrics' in result
    assert isinstance(result['audio'], torch.Tensor)

@pytest.mark.asyncio
async def test_process_chunks(audio_processor, sample_audio_file):
    audio_data = await audio_processor._load_audio(sample_audio_file)
    chunks = await audio_processor._process_chunks(audio_data)
    assert len(chunks) > 0
    assert all(isinstance(chunk['audio'], torch.Tensor) for chunk in chunks)

@pytest.mark.asyncio
async def test_noise_reduction(audio_processor, noisy_audio_file):
    result = await audio_processor.load_and_preprocess(noisy_audio_file)
    assert 'quality_metrics' in result
    assert result['quality_metrics']['signal_to_noise'] > 0

@pytest.mark.asyncio
async def test_audio_formats():
    test_formats = [
        ('test.wav', 'audio/x-wav'),
        ('test.mp3', 'audio/mpeg'),
        ('test.flac', 'audio/flac'),
        ('test.m4a', 'audio/mp4'),
        ('test.ogg', 'audio/ogg')
    ]
    
    for filename, mime_type in test_formats:
        with patch('magic.from_file', return_value=mime_type):
            assert AudioProcessor.ALLOWED_FORMATS[filename.split('.')[-1]] == [mime_type]

@pytest.mark.asyncio
async def test_quality_metrics(audio_processor, sample_audio_file):
    result = await audio_processor.load_and_preprocess(sample_audio_file)
    metrics = result['quality_metrics']
    
    assert all(metric in metrics for metric in [
        'rms_energy',
        'peak_amplitude',
        'signal_to_noise',
        'spectral_flatness',
        'spectral_bandwidth'
    ])
    assert all(isinstance(value, float) for value in metrics.values())

@pytest.mark.asyncio
async def test_memory_management(audio_processor, sample_audio_file):
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for _ in range(3):
        result = await audio_processor.load_and_preprocess(sample_audio_file)
        assert result is not None
        
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory * 1.5

@pytest.mark.asyncio
async def test_concurrent_processing(audio_processor, sample_audio_file):
    tasks = [
        audio_processor.load_and_preprocess(sample_audio_file)
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    assert all('audio' in r for r in results)

@pytest.mark.asyncio
async def test_error_handling(audio_processor):
    with pytest.raises(Exception):
        await audio_processor.load_and_preprocess('nonexistent.wav')
    
    with pytest.raises(Exception):
        await audio_processor.load_and_preprocess(None)

@pytest.mark.asyncio
async def test_sample_rate_conversion(audio_processor, sample_audio_file):
    # Create audio with different sample rate
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        orig_sample_rate = 44100
        t = np.linspace(0, duration, int(orig_sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(temp_file.name, audio, orig_sample_rate)
        
        result = await audio_processor.load_and_preprocess(temp_file.name)
        assert result['sample_rate'] == audio_processor.config.get('sample_rate', 16000)
        
    os.unlink(temp_file.name)

@pytest.mark.asyncio
async def test_channel_conversion(audio_processor):
    # Create stereo audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        stereo_audio = np.vstack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)))
        sf.write(temp_file.name, stereo_audio.T, sample_rate)
        
        result = await audio_processor.load_and_preprocess(temp_file.name)
        assert len(result['audio'].shape) == 1  # Mono
        
    os.unlink(temp_file.name)

@pytest.mark.asyncio
async def test_long_audio_processing(audio_processor):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 30.0  # 30 seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(temp_file.name, audio, sample_rate)
        
        result = await audio_processor.load_and_preprocess(temp_file.name)
        assert abs(result['duration'] - duration) < 0.1
        
    os.unlink(temp_file.name)

@pytest.mark.asyncio
async def test_silence_detection(audio_processor):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        sample_rate = 16000
        silence = np.zeros(int(sample_rate * duration))
        sf.write(temp_file.name, silence, sample_rate)
        
        result = await audio_processor.load_and_preprocess(temp_file.name)
        assert result['quality_metrics']['rms_energy'] < 0.01
        
    os.unlink(temp_file.name)

@pytest.mark.asyncio
async def test_dc_offset_detection(audio_processor):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.5  # Add DC offset
        sf.write(temp_file.name, audio, sample_rate)
        
        result = await audio_processor.load_and_preprocess(temp_file.name)
        assert 'dc_offset' in result['quality_metrics']
        
    os.unlink(temp_file.name)

@pytest.mark.asyncio
async def test_cleanup(audio_processor):
    await audio_processor.cleanup()
    assert True

if __name__ == '__main__':
    pytest.main(['-v', __file__])