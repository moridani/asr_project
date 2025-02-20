import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from unittest.mock import Mock
import os

@pytest.fixture(scope="session")
def device():
    """Global device fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def mock_cache_manager():
    """Global cache manager mock."""
    cache_manager = Mock()
    cache_manager.get_path.return_value = Path("cache/models")
    return cache_manager

@pytest.fixture
def sample_audio():
    """Generate sample audio data."""
    duration = 5
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    return torch.from_numpy(audio.astype(np.float32))

@pytest.fixture
def sample_audio_file():
    """Create temporary audio file."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        sf.write(temp_file.name, audio, sample_rate)
        yield temp_file.name
    os.unlink(temp_file.name)

@pytest.fixture
def noisy_audio_file():
    """Create temporary noisy audio file."""
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

@pytest.fixture
def multi_speaker_audio():
    """Generate multi-speaker audio data."""
    duration = 10
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    speaker1 = np.sin(2 * np.pi * 440 * t) * 0.5
    speaker2 = np.sin(2 * np.pi * 880 * t) * 0.5
    
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

@pytest.fixture
def sample_transcription():
    """Sample transcription data."""
    return {
        'segments': [
            {
                'id': '1',
                'text': 'Bonjour le monde',
                'language': 'fr',
                'start': 0.0,
                'end': 2.0
            },
            {
                'id': '2',
                'text': 'مرحبا بالعالم',
                'language': 'ar',
                'start': 2.0,
                'end': 4.0
            },
            {
                'id': '3',
                'text': 'नमस्ते दुनिया',
                'language': 'hi',
                'start': 4.0,
                'end': 6.0
            }
        ]
    }

@pytest.fixture
def diarization_result():
    """Sample diarization result."""
    return {
        'speakers': {
            'speaker_1': {'total_duration': 6.0},
            'speaker_2': {'total_duration': 4.0}
        }
    }

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Cleanup after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()