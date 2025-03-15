#!/bin/bash

# Prioritized Package Installation Script

# Function to check and install packages
install_packages() {
    echo "Starting package installation..."

    # Core Deep Learning Frameworks
    pip install torch==2.6.0+cu124 \
                torchaudio==2.6.0+cu124 \
                numpy==2.2.3 \
                --extra-index-url https://download.pytorch.org/whl/cu124

    # Machine Learning Utilities
    pip install transformers==4.38.1 \
                accelerate==1.4.0 \
                bitsandbytes==0.45.3

    # Audio Processing Libraries
    pip install faster-whisper==1.1.1 \
                pyannote.audio==3.3.2 \
                soundfile==0.13.1

    # Data Manipulation
    pip install pandas==2.2.3

    # Optional but Recommended Scientific Computing
    pip install scipy>=1.10.0 \
                scikit-learn>=1.2.0 \
                tqdm>=4.65.0

    # Development and Debugging Tools
    pip install ipython>=8.10.0 \
                jupyter>=1.0.0

    # ASR Specific Libraries
    pip install nemo-toolkit>=1.20.0 \
                pyannote.core>=5.0.0

    # Verify Installations
    echo "Verifying package installations..."
    pip list | grep -E "torch|transformers|accelerate|faster-whisper|pyannote|pandas"
}

# Verification Function
verify_installations() {
    python -c "
import torch
import transformers
import accelerate
import pyannote

print('Torch Version:', torch.__version__)
print('Transformers Version:', transformers.__version__)
print('Accelerate Version:', accelerate.__version__)
print('Pyannote Version:', pyannote.__version__)
print('CUDA Available:', torch.cuda.is_available())
"
}

# Main Execution
main() {
    install_packages
    verify_installations
}

# Run the main function
main