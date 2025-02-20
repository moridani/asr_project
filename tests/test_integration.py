import pytest
import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import json
import tempfile
import os
import shutil
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import asyncio
import time
from api.endpoints import app
from models.language_detector import LanguageDetector
from models.transcriber import Transcriber
from models.diarizer import Diarizer
from models.translator import Translator
from utils.audio_processor import AudioProcessor
from core.pipeline import ASRPipeline

class TestIntegrationFlow:
    @pytest.fixture(autouse=True)
    def setup_dirs(self):
        """Create and cleanup test directories."""
        test_dirs = ['test_cache', 'test_uploads', 'test_results', 'test_models']
        for dir_name in test_dirs:
            os.makedirs(dir_name, exist_ok=True)
        yield
        for dir_name in test_dirs:
            shutil.rmtree(dir_name, ignore_errors=True)

    @pytest.fixture
    def audio_files(self):
        """Generate test audio files."""
        files = {}
        sample_rate = 16000
        
        # Generate different durations of audio
        durations = {
            'short': 2.0,
            'medium': 10.0,
            'long': 30.0
        }
        
        for name, duration in durations.items():
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
                sf.write(temp_file.name, audio, sample_rate)
                files[name] = temp_file.name
        
        yield files
        
        for file_path in files.values():
            if os.path.exists(file_path):
                os.unlink(file_path)

    @pytest.fixture
    def pipeline(self):
        """Initialize ASR pipeline."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return ASRPipeline(device=device)

    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, audio_files, pipeline):
        """Test complete processing pipeline."""
        client = TestClient(app)
        
        # File upload
        with open(audio_files['medium'], 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.wav", f, "audio/wav")}
            )
        assert response.status_code == 200
        file_id = response.json()['file_id']
        
        # Process request
        response = client.post(
            "/process",
            json={
                "file_id": file_id,
                "config": {
                    "languages": ["fr", "ar", "hi"],
                    "min_speakers": 1,
                    "max_speakers": 5
                }
            }
        )
        assert response.status_code == 200
        task_id = response.json()['task_id']
        
        # Monitor processing
        for _ in range(30):  # 30 second timeout
            response = client.get(f"/status/{task_id}")
            assert response.status_code == 200
            status = response.json()
            if status['status'] == 'completed':
                break
            elif status['status'] == 'failed':
                pytest.fail("Processing failed")
            await asyncio.sleep(1)
        
        # Get results
        response = client.get(f"/result/{task_id}")
        assert response.status_code == 200
        result = response.json()
        
        # Validate result structure
        self._validate_result_structure(result)
        
        # Cleanup
        response = client.delete(f"/task/{task_id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_error_handling(self, audio_files):
        """Test error handling scenarios."""
        client = TestClient(app)
        
        # Invalid file format
        with tempfile.NamedTemporaryFile(suffix='.txt') as invalid_file:
            invalid_file.write(b"Not an audio file")
            invalid_file.seek(0)
            response = client.post(
                "/upload",
                files={"file": ("test.txt", invalid_file, "text/plain")}
            )
            assert response.status_code == 400
        
        # Invalid file ID
        response = client.post(
            "/process",
            json={"file_id": "nonexistent"}
        )
        assert response.status_code == 404
        
        # Invalid configuration
        with open(audio_files['short'], 'rb') as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.wav", f, "audio/wav")}
            )
            file_id = upload_response.json()['file_id']
            
            response = client.post(
                "/process",
                json={
                    "file_id": file_id,
                    "config": {
                        "min_speakers": 5,
                        "max_speakers": 2
                    }
                }
            )
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, audio_files):
        """Test concurrent processing capabilities."""
        client = TestClient(app)
        num_concurrent = 3
        task_ids = []
        
        # Start concurrent tasks
        with open(audio_files['short'], 'rb') as f:
            for _ in range(num_concurrent):
                upload_response = client.post(
                    "/upload",
                    files={"file": ("test.wav", f, "audio/wav")}
                )
                file_id = upload_response.json()['file_id']
                
                process_response = client.post(
                    "/process",
                    json={"file_id": file_id}
                )
                task_ids.append(process_response.json()['task_id'])
        
        # Monitor all tasks
        completed = 0
        for _ in range(30):  # 30 second timeout
            completed = sum(
                1 for task_id in task_ids
                if client.get(f"/status/{task_id}").json()['status'] == 'completed'
            )
            if completed == num_concurrent:
                break
            await asyncio.sleep(1)
        
        assert completed == num_concurrent

    @pytest.mark.asyncio
    async def test_memory_management(self, audio_files):
        """Test memory usage under load."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        client = TestClient(app)
        initial_memory = torch.cuda.memory_allocated()
        
        # Process large file multiple times
        with open(audio_files['long'], 'rb') as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.wav", f, "audio/wav")}
            )
            file_id = upload_response.json()['file_id']
        
        for _ in range(3):
            process_response = client.post(
                "/process",
                json={"file_id": file_id}
            )
            task_id = process_response.json()['task_id']
            
            while client.get(f"/status/{task_id}").json()['status'] != 'completed':
                await asyncio.sleep(1)
        
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory * 2

    def _validate_result_structure(self, result):
        """Validate result structure and content."""
        # Check main sections
        assert all(key in result for key in ['segments', 'translations', 'metadata'])
        
        # Validate segments
        for segment in result['segments']:
            assert all(key in segment for key in [
                'start_time',
                'end_time',
                'text',
                'speaker',
                'language',
                'confidence'
            ])
            assert 0 <= segment['confidence'] <= 1
            assert segment['start_time'] < segment['end_time']
        
        # Validate translations
        for translation in result['translations'].values():
            assert all(key in translation for key in [
                'text',
                'confidence',
                'model_used'
            ])
            assert 0 <= translation['confidence'] <= 1
        
        # Validate metadata
        assert 'processing_time' in result['metadata']
        assert result['metadata']['processing_time'] > 0

    @pytest.mark.asyncio
    async def test_long_audio_processing(self, audio_files):
        """Test processing of long audio files."""
        client = TestClient(app)
        
        with open(audio_files['long'], 'rb') as f:
            upload_response = client.post(
                "/upload",
                files={"file": ("test.wav", f, "audio/wav")}
            )
            file_id = upload_response.json()['file_id']
        
        process_response = client.post(
            "/process",
            json={"file_id": file_id}
        )
        task_id = process_response.json()['task_id']
        
        # Monitor with progress tracking
        previous_progress = 0
        for _ in range(60):  # 60 second timeout
            status = client.get(f"/status/{task_id}").json()
            current_progress = status['progress']
            assert current_progress >= previous_progress
            previous_progress = current_progress
            
            if status['status'] == 'completed':
                break
            await asyncio.sleep(1)
        
        result = client.get(f"/result/{task_id}").json()
        assert len(result['segments']) > 10  # Expect multiple segments

    @pytest.mark.asyncio
    async def test_multilingual_processing(self, pipeline):
        """Test processing of multilingual content."""
        # Create multilingual audio file
        sample_rate = 16000
        duration = 15.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            result = await pipeline.process_file(temp_file.name)
        
        os.unlink(temp_file.name)
        
        # Validate language detection
        languages = set(seg['language'] for seg in result['segments'])
        assert len(languages) > 0
        
        # Validate translations
        for translation in result['translations'].values():
            assert translation['confidence'] > 0.5

    @pytest.mark.asyncio
    async def test_api_performance(self, audio_files):
        """Test API performance metrics."""
        client = TestClient(app)
        
        start_time = time.time()
        with open(audio_files['medium'], 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.wav", f, "audio/wav")}
            )
        upload_time = time.time() - start_time
        assert upload_time < 5.0  # Upload should be quick
        
        file_id = response.json()['file_id']
        process_response = client.post(
            "/process",
            json={"file_id": file_id}
        )
        task_id = process_response.json()['task_id']
        
        processing_start = time.time()
        while True:
            status = client.get(f"/status/{task_id}").json()
            if status['status'] == 'completed':
                break
            assert time.time() - processing_start < 300  # Max 5 minutes
            await asyncio.sleep(1)

if __name__ == '__main__':
    pytest.main(['-v', __file__])