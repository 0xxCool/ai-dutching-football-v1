#!/bin/bash
# Football AI System - Phase 1: Python Environment & ML Stack Setup
# Optimiert für Ubuntu 24.04 LTS mit RTX 3090 und CUDA 12.7

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# System checks
check_system_requirements() {
    log "🔍 Überprüfe System-Voraussetzungen..."
    
    # Check if running as footballai user
    if [ "$USER" != "footballai" ]; then
        error "Bitte als footballai Benutzer ausführen (use: su - footballai)"
        exit 1
    fi
    
    # Check system resources
    TOTAL_MEM=$(free -g | awk 'NR==2{printf "%.1f", $2}')
    if (( $(echo "$TOTAL_MEM < 32" | bc -l) )); then
        warn "Weniger als 32GB RAM verfügbar: ${TOTAL_MEM}GB"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h / | awk 'NR==2 {print $4}')
    log "💾 Verfügbarer Speicherplatz: $DISK_SPACE"
    
    # GPU Status
    if command -v nvidia-smi &> /dev/null; then
        log "✅ NVIDIA GPU erkannt:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        error "Keine NVIDIA GPU erkannt!"
        exit 1
    fi
    
    # CUDA Status
    if command -v nvcc &> /dev/null; then
        log "✅ CUDA installiert:"
        nvcc --version
    else
        error "CUDA nicht installiert!"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    log "📦 Installiere System-Abhängigkeiten..."
    
    sudo apt update && sudo apt upgrade -y
    
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        pkg-config \
        libhdf5-dev \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libcanberra-gtk* \
        libgdk-pixbuf2.0-dev \
        libpango1.0-dev \
        libcairo2-dev \
        libhdf5-serial-dev \
        libhdf5-dev \
        libhdf5-openmpi-dev \
        python3-dev \
        python3-pip \
        python3-numpy \
        python3-matplotlib \
        python3-scipy \
        python3-pandas \
        python3-sklearn \
        python3-venv \
        python3-opencv \
        libtesseract-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu
}

# Setup Python environment
setup_python_env() {
    log "🐍 Richte Python Environment ein..."
    
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "football-ai"; then
        log "Erstelle Conda Environment..."
        conda create -n football-ai python=3.10 -y
    fi
    
    # Activate environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate football-ai
    
    # Upgrade pip and basic tools
    pip install --upgrade pip setuptools wheel
    pip install --upgrade pip-tools
    
    log "✅ Python Environment aktiviert"
}

# Install PyTorch with CUDA support
install_pytorch() {
    log "🔥 Installiere PyTorch mit CUDA 12.7 Support..."
    
    # Uninstall any existing PyTorch installations
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    
    # Install PyTorch with CUDA 12.1 (compatible with 12.7)
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify PyTorch installation
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA verfügbar: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
    
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log "✅ PyTorch mit GPU Support erfolgreich installiert"
    else
        error "PyTorch GPU Support fehlgeschlagen!"
        exit 1
    fi
}

# Install TensorFlow
install_tensorflow() {
    log "🧠 Installiere TensorFlow mit GPU Support..."
    
    # Uninstall existing TensorFlow
    pip uninstall tensorflow tensorflow-gpu -y 2>/dev/null || true
    
    # Install TensorFlow with GPU support
    pip install tensorflow==2.14.0
    
    # Verify TensorFlow installation
    python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU verfügbar: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
    
    if python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')) > 0)" | grep -q "True"; then
        log "✅ TensorFlow mit GPU Support erfolgreich installiert"
    else
        warn "TensorFlow GPU Support nicht verfügbar"
    fi
}

# Install JAX
install_jax() {
    log "⚡ Installiere JAX mit GPU Support..."
    
    # Install JAX with CUDA support
    pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install jaxlib
    
    # Verify JAX installation
    python -c "import jax; print(f'JAX: {jax.__version__}'); print(f'GPU verfügbar: {jax.devices()[0].platform}')"
    
    log "✅ JAX installiert"
}

# Install ML and Data Science libraries
install_ml_libraries() {
    log "📊 Installiere ML und Data Science Bibliotheken..."
    
    # Hugging Face libraries
    pip install transformers==4.35.0 \
                datasets \
                accelerate \
                evaluate \
                tokenizers \
                huggingface-hub
    
    # Traditional ML libraries
    pip install scikit-learn==1.3.2 \
                dask[complete] \
                xgboost==1.7.6 \
                lightgbm==3.3.5 \
                catboost==1.2.2 \
                optuna==3.4.0 \
                hyperopt==0.2.7 \
                scikit-optimize==0.9.0
    
    # Data science libraries
    pip install pandas==2.1.3 \
                numpy==1.25.2 \
                scipy==1.11.4 \
                matplotlib==3.8.1 \
                seaborn==0.13.0 \
                plotly==5.18.0 \
                jupyter \
                jupyterlab \
                ipython
    
    # Computer vision
    pip install opencv-python==4.8.1.78 \
                pillow==10.1.0 \
                albumentations==1.3.1 \
                timm==0.9.12 \
                imageio \
                scikit-image
    
    # NLP libraries
    pip install spacy==3.7.2 \
                nltk==3.8.1 \
                gensim==4.3.2 \
                textblob \
                wordcloud
    
    # Download spaCy model
    python -m spacy download en_core_web_sm
    
    log "✅ ML Bibliotheken installiert"
}

# Install API and Web frameworks
install_web_frameworks() {
    log "🌐 Installiere Web Frameworks..."
    
    pip install fastapi==0.104.1 \
                uvicorn[standard]==0.24.0 \
                flask==3.0.0 \
                django==4.2.7 \
                starlette \
                pydantic \
                python-multipart \
                aiofiles
    
    # Database libraries
    pip install sqlalchemy==2.0.23 \
                alembic \
                psycopg2-binary \
                asyncpg \
                redis \
                aioredis \
                pymongo \
                motor
    
    # Authentication & Security
    pip install python-jose[cryptography] \
                passlib[bcrypt] \
                python-multipart \
                cryptography \
                PyJWT
    
    # HTTP clients
    pip install requests \
                httpx \
                aiohttp \
                websockets \
                python-socketio
    
    log "✅ Web Frameworks installiert"
}

# Install additional tools
install_additional_tools() {
    log "🔧 Installiere zusätzliche Tools..."
    
    # Development tools
    pip install pytest \
                pytest-asyncio \
                pytest-cov \
                black \
                flake8 \
                mypy \
                pre-commit \
                isort
    
    # Monitoring and logging
    pip install prometheus-client \
                grafana-api \
                structlog \
                python-json-logger \
                sentry-sdk
    
    # Utilities
    pip install python-dotenv \
                pyyaml \
                toml \
                click \
                typer \
                rich \
                tqdm \
                schedule \
                APScheduler
    
    # Data processing
    pip install polars \
                modin[all] \
                vaex \
                pyarrow \
                fastparquet
    
    # Visualization
    pip install bokeh \
                altair \
                streamlit \
                gradio
    
    log "✅ Zusätzliche Tools installiert"
}

# Verify installation
verify_installation() {
    log "🔍 Überprüfe Installation..."
    
    # Create verification script
    cat > verify_installation.py << 'EOF'
import sys
import importlib

required_packages = [
    'torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'sklearn',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'fastapi',
    'uvicorn', 'sqlalchemy', 'redis', 'transformers', 'datasets'
]

print("=== Installations-Überprüfung ===")
print(f"Python Version: {sys.version}")
print()

for package in required_packages:
    try:
        module = importlib.import_module(package)
        if hasattr(module, '__version__'):
            print(f"✅ {package}: {module.__version__}")
        else:
            print(f"✅ {package}: Version nicht verfügbar")
    except ImportError as e:
        print(f"❌ {package}: Fehler - {e}")

# GPU Check
try:
    import torch
    print(f"\nGPU verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"GPU Check Fehler: {e}")

print("\n=== Überprüfung abgeschlossen ===")
EOF
    
    python verify_installation.py
    rm verify_installation.py
}

# Main execution
main() {
    log "🚀 Starte Football AI Python Environment Setup"
    log "📋 System: Ubuntu 24.04 LTS + RTX 3090 + CUDA 12.7"
    
    check_system_requirements
    install_system_deps
    setup_python_env
    
    # Install ML frameworks
    install_pytorch
    install_tensorflow
    install_jax
    install_ml_libraries
    
    # Install web frameworks and tools
    install_web_frameworks
    install_additional_tools
    
    # Final verification
    verify_installation
    
    log "✅ Python Environment Setup abgeschlossen!"
    log "🎯 Umgebung aktivieren: conda activate football-ai"
    log "📁 Nächster Schritt: ./02-project-setup.sh"
}

# Execute main function
main "$@"