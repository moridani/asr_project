# Multilingual Speech Recognition System

An advanced Automatic Speech Recognition (ASR) system with multilingual support, focusing on French language accuracy and speaker diarization.

## Key Features

- **Language Detection**: Accurate identification of spoken languages with French optimization
- **Speech Recognition**: High-accuracy transcription using state-of-the-art models
- **Speaker Diarization**: Precise speaker separation with overlap detection
- **Translation**: Specialized processing for French-to-English translation
- **Memory Optimization**: Efficient resource usage for processing long audio files
- **Progress Tracking**: Real-time updates during processing

## Components

### 1. Language Detector

Enhanced language detection with multiple model support:

- **VoxLingua107**: Primary model for language identification
- **Whisper**: Secondary model for cross-validation
- **French-specific Model**: Additional optimization for French language
- **Segmentation**: Advanced timeline segmentation for language changes
- **Memory Management**: Efficient processing of large audio files

### 2. Transcriber

Enhanced transcription with optimized French language support:

- **Whisper Large-v3**: Primary model for transcription
- **Conformer**: Backup model for fallback processing
- **French Wav2Vec2**: Specialized model for French content
- **Custom Vocabulary**: Domain-specific term handling
- **Batch Processing**: Efficient memory management for long files

### 3. Diarizer

Enhanced speaker diarization with improved clustering:

- **PyAnnote**: Primary diarization model
- **Multiple Embedding Types**: Combined ECAPA-TDNN and Resemblyzer
- **Overlap Detection**: Identifies regions with multiple speakers
- **Turn Analysis**: Recognizes conversation patterns
- **Speaker Metrics**: Provides speaker balance and characteristics

### 4. Translator

Specialized translation for multilingual content:

- **French-to-English**: Optimized for French content
- **NLLB Model**: Support for 200+ languages
- **Context-aware**: Maintains speaker context across translations

## System Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum (32GB+ recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for large models)
- **Storage**: 20GB+ for models and cache
- **Python**: 3.8+
- **Dependencies**: PyTorch, Transformers, faster-whisper, pyannote-audio, etc.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/asr-system.git
cd asr-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up authentication (if needed):
Copy the `.env.template` file to `.env` and update with your HuggingFace token:
```bash
cp config/.env.template config/.env
# Edit .env with your settings
```

## Usage

### Command Line Interface

#### Process a single file:

```bash
python main.py process_file --input-file path/to/audio.wav --output-file results.json
```

#### Process multiple files in a batch:

```bash
python main.py batch_process --input-dir path/to/files --output-dir path/to/results
```

#### Start the API server:

```bash
python main.py start_server --host 0.0.0.0 --port 8000
```

### API Usage

The system provides a RESTful API for processing audio files:

#### Upload a file:

```python
import requests

# Upload file
with open('audio.wav', 'rb') as f:
    response = requests.post('http://localhost:8000/upload', files={'file': f})
    
file_id = response.json()['file_id']
```

#### Process the file:

```python
# Create processing request
config = {
    'min_speakers': 1,
    'max_speakers': 5,
    'language_detection_confidence': 0.6
}

response = requests.post(
    'http://localhost:8000/process',
    json={'file_id': file_id, 'config': config}
)

task_id = response.json()['task_id']
```

#### Get results:

```python
# Check status
status_response = requests.get(f'http://localhost:8000/status/{task_id}')
status = status_response.json()

# If completed, get results
if status['status'] == 'completed':
    result = requests.get(f'http://localhost:8000/result/{task_id}')
    transcription = result.json()
```

## Configuration

### Main Configuration Options

The system is highly configurable through `config/settings.py`:

```python
# Hardware configuration
DEVICE = "cuda"  # or "cpu"
NUM_THREADS = 4
BATCH_SIZE = 8

# Model selection
WHISPER_MODEL = "large-v3"
FRENCH_MODEL = "facebook/wav2vec2-large-xlsr-53-french"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Processing parameters
MIN_SPEAKERS = 1
MAX_SPEAKERS = 10
SAMPLE_RATE = 16000
```

### Memory Optimization

For systems with limited memory, adjust these settings:

```python
# Reduce memory usage
WHISPER_MODEL = "medium"  # Use smaller model
BATCH_SIZE = 4  # Process smaller batches
MAX_CACHE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB cache limit
```

### French Language Optimization

The system is optimized for French language content:

```python
# French optimization settings
OPTIMIZE_FOR_FRENCH = True
FRENCH_MODEL = "facebook/wav2vec2-large-xlsr-53-french"
```

## Enhanced Components

### Language Detector Enhancements

- **Memory Optimization**: Controlled memory usage with cleanup routines
- **French Detection**: Special handling for French language content
- **Segment Refinement**: Improved boundary detection for language changes
- **Progress Tracking**: Real-time progress updates during processing

### Transcriber Enhancements

- **Model Selection**: Intelligent selection based on detected language
- **Segment Processing**: Memory-efficient batch processing
- **Vocabulary Enhancement**: Custom vocabulary for domain-specific terms
- **French Optimization**: Specialized handling for French pronunciation

### Diarizer Enhancements

- **Speaker Clustering**: Improved algorithm for speaker identification
- **Overlap Detection**: Enhanced detection of overlapping speech
- **Speaker Characteristics**: Analysis of speaking patterns
- **Confidence Scoring**: Reliability metrics for speaker assignment

## Performance Optimization

### GPU Acceleration

The system is optimized for GPU processing:

```python
# GPU optimization
DEVICE = "cuda"
GPU_MEMORY_FRACTION = 0.9
```

### CPU Optimization

For CPU-only systems:

```python
# CPU optimization
DEVICE = "cpu"
NUM_THREADS = 8  # Adjust to your CPU core count
USE_INT8_QUANTIZATION = True  # Use quantized models
```

### Memory Management

The system includes advanced memory management:

- Garbage collection after processing steps
- Configurable cache size and cleanup
- Batch processing for large files
- Model unloading when not in use

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce `BATCH_SIZE` in settings
   - Use a smaller Whisper model
   - Enable quantization for reduced memory usage

2. **Slow Processing**:
   - Ensure GPU is being utilized (`DEVICE="cuda"`)
   - Increase `BATCH_SIZE` if memory allows
   - Adjust `NUM_THREADS` for CPU processing

3. **Inaccurate Diarization**:
   - Adjust `MIN_SPEAKERS` and `MAX_SPEAKERS`
   - Increase `OVERLAP_THRESHOLD` for better detection
   - Check audio quality and consider pre-processing

## Contributors

- Your Name <your.email@example.com>

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper
- PyAnnote Audio
- SpeechBrain
- Hugging Face Transformers