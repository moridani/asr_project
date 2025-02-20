import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json
import asyncio
import aiohttp
from unittest.mock import Mock, patch
import tempfile
import shutil
import os
from api.endpoints import app, UPLOAD_DIR, RESULTS_DIR

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def setup_test_dirs():
    temp_upload = Path(tempfile.mkdtemp())
    temp_results = Path(tempfile.mkdtemp())
    
    original_upload = UPLOAD_DIR
    original_results = RESULTS_DIR
    
    global UPLOAD_DIR, RESULTS_DIR
    UPLOAD_DIR = temp_upload
    RESULTS_DIR = temp_results
    
    yield
    
    shutil.rmtree(temp_upload)
    shutil.rmtree(temp_results)
    
    UPLOAD_DIR = original_upload
    RESULTS_DIR = original_results

@pytest.fixture
def sample_audio_file():
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(b'dummy audio data')
        return temp_file.name

@pytest.fixture
def mock_pipeline():
    with patch('api.endpoints.pipeline') as mock:
        mock.process_file.return_value = {
            'segments': [
                {
                    'text': 'Test transcription',
                    'start': 0.0,
                    'end': 1.0,
                    'speaker': 'speaker_1'
                }
            ],
            'translations': {
                'fr': 'Test translation'
            }
        }
        yield mock

@pytest.mark.asyncio
async def test_upload_endpoint(client, setup_test_dirs, sample_audio_file):
    with open(sample_audio_file, 'rb') as f:
        response = client.post(
            "/upload",
            files={"file": ("test.wav", f, "audio/wav")}
        )
    assert response.status_code == 200
    assert 'file_id' in response.json()

@pytest.mark.asyncio
async def test_process_endpoint(client, setup_test_dirs, mock_pipeline):
    response = client.post(
        "/process",
        json={
            "file_id": "test_id",
            "config": {"language": "en"}
        }
    )
    assert response.status_code == 200
    assert 'task_id' in response.json()

@pytest.mark.asyncio
async def test_status_endpoint(client, setup_test_dirs):
    # Setup test task
    task_id = "test_task"
    app.state.active_tasks = {
        task_id: {
            'status': 'processing',
            'progress': 0.5
        }
    }
    
    response = client.get(f"/status/{task_id}")
    assert response.status_code == 200
    assert response.json()['progress'] == 0.5

@pytest.mark.asyncio
async def test_result_endpoint(client, setup_test_dirs):
    task_id = "test_task"
    result_path = RESULTS_DIR / f"{task_id}.json"
    result_data = {"test": "result"}
    
    # Setup test result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(result_data, f)
    
    app.state.active_tasks = {
        task_id: {'status': 'completed'}
    }
    
    response = client.get(f"/result/{task_id}")
    assert response.status_code == 200
    assert response.json() == result_data

@pytest.mark.asyncio
async def test_delete_task(client, setup_test_dirs):
    task_id = "test_task"
    file_id = "test_file"
    
    # Setup test files
    app.state.active_tasks = {
        task_id: {
            'status': 'completed',
            'file_id': file_id
        }
    }
    
    result_path = RESULTS_DIR / f"{task_id}.json"
    with open(result_path, 'w') as f:
        json.dump({}, f)
        
    upload_path = UPLOAD_DIR / f"{file_id}.wav"
    with open(upload_path, 'w') as f:
        f.write('test')
    
    response = client.delete(f"/task/{task_id}")
    assert response.status_code == 200
    assert not result_path.exists()
    assert not upload_path.exists()

@pytest.mark.asyncio
async def test_error_handling(client, setup_test_dirs):
    # Test invalid file upload
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"invalid", "text/plain")}
    )
    assert response.status_code == 400
    
    # Test invalid process request
    response = client.post(
        "/process",
        json={"file_id": "nonexistent"}
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_concurrent_requests(client, setup_test_dirs, mock_pipeline):
    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://testserver/process",
                json={"file_id": "test_id"}
            ) as response:
                return await response.json()
    
    tasks = [make_request() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all('task_id' in r for r in results)

@pytest.mark.asyncio
async def test_progress_tracking(client, setup_test_dirs):
    task_id = "test_task"
    
    # Simulate progress updates
    progress_values = [0.0, 0.5, 1.0]
    for progress in progress_values:
        app.state.active_tasks[task_id] = {
            'status': 'processing' if progress < 1 else 'completed',
            'progress': progress
        }
        
        response = client.get(f"/status/{task_id}")
        assert response.status_code == 200
        assert response.json()['progress'] == progress

@pytest.mark.asyncio
async def test_webhook_notifications(client, setup_test_dirs, mock_pipeline):
    webhook_server = Mock()
    webhook_url = "http://test-webhook.com/callback"
    
    response = client.post(
        "/process",
        json={
            "file_id": "test_id",
            "webhook_url": webhook_url
        }
    )
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_resource_cleanup(client, setup_test_dirs):
    # Test cleanup on server shutdown
    with patch('api.endpoints.cleanup_resources') as mock_cleanup:
        await app.router.shutdown()
        mock_cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_large_file_handling(client, setup_test_dirs):
    large_file_size = 100 * 1024 * 1024  # 100MB
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
        temp_file.write(b'0' * large_file_size)
        temp_file.seek(0)
        
        response = client.post(
            "/upload",
            files={"file": ("large.wav", temp_file, "audio/wav")}
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_api_rate_limiting():
    # Test rate limiting middleware if implemented
    pass

@pytest.mark.asyncio
async def test_api_authentication():
    # Test authentication middleware if implemented
    pass

@pytest.mark.asyncio
async def test_server_startup(setup_test_dirs):
    event = {'called': False}
    
    async def startup():
        event['called'] = True
    
    await app.router.startup()
    assert event['called']

if __name__ == '__main__':
    pytest.main(['-v', __file__])