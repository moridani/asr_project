import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio
from models.translator import Translator
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
async def translator(device, mock_cache_manager):
    translator = Translator(device, mock_cache_manager)
    yield translator
    await translator.cleanup()

@pytest.fixture
def sample_transcription():
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
    return {
        'speakers': {
            'speaker_1': {'total_duration': 6.0},
            'speaker_2': {'total_duration': 4.0}
        }
    }

@pytest.mark.asyncio
async def test_initialization(translator):
    assert translator.fr_model is not None
    assert translator.nllb_model is not None
    assert hasattr(translator, 'thread_pool')

@pytest.mark.asyncio
async def test_french_translation(translator):
    french_text = ["Bonjour le monde", "Comment allez-vous"]
    translations = await translator._translate_french(french_text)
    assert len(translations) == 2
    assert all(isinstance(t, str) for t in translations)
    assert all(len(t) > 0 for t in translations)

@pytest.mark.asyncio
async def test_nllb_translation(translator):
    arabic_text = ["مرحبا بالعالم", "كيف حالك"]
    translations = await translator._translate_nllb(arabic_text, 'ar')
    assert len(translations) == 2
    assert all(isinstance(t, str) for t in translations)

@pytest.mark.asyncio
async def test_batch_translation(translator, sample_transcription, diarization_result):
    result = await translator.translate_segments(sample_transcription, diarization_result)
    assert 'translations' in result
    assert 'metadata' in result
    assert len(result['translations']) == len(sample_transcription['segments'])

@pytest.mark.asyncio
async def test_language_specific_parameters(translator):
    params = translator._get_translation_parameters('fr')
    assert all(key in params for key in ['max_length', 'num_beams', 'length_penalty', 'temperature'])
    
    # Test different languages
    languages = ['fr', 'ar', 'hi', 'zh']
    for lang in languages:
        params = translator._get_translation_parameters(lang)
        assert params['num_beams'] > 0
        assert 0 < params['temperature'] <= 1

@pytest.mark.asyncio
async def test_error_handling(translator):
    with pytest.raises(Exception):
        await translator.translate_segments(None, None)
    
    with pytest.raises(Exception):
        await translator.translate_segments({'segments': []}, None)

@pytest.mark.asyncio
async def test_concurrent_translation(translator, sample_transcription, diarization_result):
    tasks = [
        translator.translate_segments(sample_transcription, diarization_result)
        for _ in range(3)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    assert all('translations' in r for r in results)

@pytest.mark.asyncio
async def test_memory_management(translator, sample_transcription, diarization_result):
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for _ in range(3):
        result = await translator.translate_segments(sample_transcription, diarization_result)
        assert result is not None
        
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert final_memory <= initial_memory * 1.5

@pytest.mark.asyncio
async def test_long_text_handling(translator):
    long_text = " ".join(["Bonjour le monde"] * 100)  # Very long text
    segments = [{'id': '1', 'text': long_text, 'language': 'fr'}]
    result = await translator.translate_segments(
        {'segments': segments},
        {'speakers': {}}
    )
    assert result['translations']['1']['text'] is not None

@pytest.mark.asyncio
async def test_language_confidence(translator, sample_transcription, diarization_result):
    result = await translator.translate_segments(sample_transcription, diarization_result)
    
    for translation in result['translations'].values():
        assert 'confidence' in translation
        assert 0 <= translation['confidence'] <= 1

@pytest.mark.asyncio
async def test_model_selection(translator, sample_transcription, diarization_result):
    result = await translator.translate_segments(sample_transcription, diarization_result)
    
    for translation in result['translations'].values():
        assert 'model_used' in translation
        assert translation['model_used'] in ['helsinki-fr', 'nllb']

@pytest.mark.asyncio
async def test_special_characters(translator):
    special_chars_text = "Hello! @#$%^&* World"
    segments = [{'id': '1', 'text': special_chars_text, 'language': 'en'}]
    result = await translator.translate_segments(
        {'segments': segments},
        {'speakers': {}}
    )
    assert result['translations']['1']['text'] is not None

@pytest.mark.asyncio
async def test_empty_text(translator):
    empty_text = ""
    segments = [{'id': '1', 'text': empty_text, 'language': 'fr'}]
    result = await translator.translate_segments(
        {'segments': segments},
        {'speakers': {}}
    )
    assert '1' in result['translations']

@pytest.mark.asyncio
async def test_unknown_language(translator):
    unknown_lang_text = "Test text"
    segments = [{'id': '1', 'text': unknown_lang_text, 'language': 'xx'}]
    result = await translator.translate_segments(
        {'segments': segments},
        {'speakers': {}}
    )
    assert result['translations']['1']['confidence'] < 0.8

@pytest.mark.asyncio
async def test_translation_quality_metrics(translator, sample_transcription, diarization_result):
    result = await translator.translate_segments(sample_transcription, diarization_result)
    
    assert 'metadata' in result
    assert 'processing_time' in result['metadata']
    assert 'languages_processed' in result['metadata']
    assert isinstance(result['metadata']['languages_processed'], list)

@pytest.mark.asyncio
async def test_performance_benchmarking(translator, sample_transcription, diarization_result):
    start_time = time.time()
    result = await translator.translate_segments(sample_transcription, diarization_result)
    processing_time = time.time() - start_time
    
    assert result['metadata']['processing_time'] <= processing_time
    assert isinstance(result['metadata']['processing_time'], (int, float))

@pytest.mark.asyncio
async def test_batch_size_handling(translator):
    # Test with large number of segments
    many_segments = [
        {
            'id': str(i),
            'text': f'Test text {i}',
            'language': 'fr' if i % 2 == 0 else 'ar'
        }
        for i in range(20)
    ]
    
    result = await translator.translate_segments(
        {'segments': many_segments},
        {'speakers': {}}
    )
    assert len(result['translations']) == len(many_segments)

@pytest.mark.asyncio
async def test_cleanup(translator):
    await translator.cleanup()
    assert True

if __name__ == '__main__':
    pytest.main(['-v', __file__])