#!/bin/bash
# Football AI System - Phase 2: Project Structure & Configuration Setup
# Erstellt vollständige Projektstruktur und Konfiguration

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

# Check if running as footballai user
check_user() {
    if [ "$USER" != "footballai" ]; then
        error "Bitte als footballai Benutzer ausführen (use: su - footballai)"
        exit 1
    fi
}

# Create project directory structure
create_project_structure() {
    log "🏗️ Erstelle Projekt-Verzeichnisstruktur..."
    
    # Main project directory
    mkdir -p ~/football-ai-system/{backend,frontend,data,models,config,scripts,docs,tests,logs}
    
    # Backend structure
    mkdir -p ~/football-ai-system/backend/{app,models,services,utils,tests,core,db,schemas}
    mkdir -p ~/football-ai-system/backend/app/{api,core,db,schemas}
    mkdir -p ~/football-ai-system/backend/api/{v1,endpoints}
    mkdir -p ~/football-ai-system/backend/core/{config,security,middleware}
    mkdir -p ~/football-ai-system/backend/db/{repositories,sessions}
    mkdir -p ~/football-ai-system/backend/schemas/{requests,responses}
    
    # Frontend structure
    mkdir -p ~/football-ai-system/frontend/{src,public,components,pages,utils,services,tests}
    mkdir -p ~/football-ai-system/frontend/src/{components,pages,hooks,services,utils,types,styles}
    mkdir -p ~/football-ai-system/frontend/components/{dashboard,predictions,dutching,matches,common}
    mkdir -p ~/football-ai-system/frontend/src/services/{api,websocket,storage}
    mkdir -p ~/football-ai-system/frontend/src/hooks/{auth,data,ui}
    
    # Data structure
    mkdir -p ~/football-ai-system/data/{raw,processed,features,models,backtests,exports,logs}
    mkdir -p ~/football-ai-system/data/raw/{matches,teams,players,odds,leagues,statistics}
    mkdir -p ~/football-ai-system/data/processed/{features,training,validation,test}
    mkdir -p ~/football-ai-system/data/features/{engineered,selected,transformed}
    
    # Models structure
    mkdir -p ~/football-ai-system/models/{neural_nets,ensemble,traditional,pretrained,checkpoints,exports}
    mkdir -p ~/football-ai-system/models/neural_nets/{correct_score,over_under,btts,match_winner}
    mkdir -p ~/football-ai-system/models/ensemble/{stacking,blending,voting,averaging}
    mkdir -p ~/football-ai-system/models/traditional/{xgboost,lightgbm,catboost,logistic_regression,random_forest,svm}
    
    # Config structure
    mkdir -p ~/football-ai-system/config/{envs,templates,models,database,logging,nginx}
    
    # Scripts structure
    mkdir -p ~/football-ai-system/scripts/{setup,data,training,deployment,maintenance,monitoring}
    
    # Documentation
    mkdir -p ~/football-ai-system/docs/{api,architecture,deployment,usage,development}
    
    log "✅ Projekt-Verzeichnisstruktur erstellt"
}

# Create .env file
create_env_file() {
    log "📝 Erstelle .env Konfigurationsdatei..."
    
    cat > ~/football-ai-system/.env << 'EOF'
# Football AI System Configuration - Production Ready
# Achtung: Diese Datei NICHT in Git committen!

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DEVELOPMENT_MODE=true

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# PostgreSQL Configuration
DATABASE_URL=postgresql://football_ai:football_ai_password@localhost:5432/football_ai_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=football_ai_db
POSTGRES_USER=football_ai
POSTGRES_PASSWORD=football_ai_password

# TimescaleDB Configuration
TIMESCALE_HOST=localhost
TIMESCALE_PORT=5433
TIMESCALE_DB=football_ai_timescale
TIMESCALE_USER=football_ai
TIMESCALE_PASSWORD=football_ai_password

# Redis Configuration
REDIS_URL=redis://:redis_password@localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password
REDIS_DB=0

# =============================================================================
# API KEYS & EXTERNAL SERVICES
# =============================================================================
# Football Data APIs (Ersetzen mit echten Keys!)
SPORTMONKS_API_KEY=your_sportmonks_api_key_here
FOOTBALL_DATA_API_KEY=your_football_data_api_key_here
ODDS_API_KEY=your_odds_api_key_here
WEATHER_API_KEY=your_weather_api_key_here

# Other Services
SENTRY_DSN=your_sentry_dsn_here
GRAFANA_API_KEY=your_grafana_api_key_here

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================
# JWT Configuration
SECRET_KEY=your_super_secret_key_change_this_in_production
JWT_SECRET=your_jwt_secret_change_this_in_production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_MINUTES=1440

# Password Hashing
PASSWORD_HASH_ALGORITHM=bcrypt
BCRYPT_ROUNDS=12

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
ALLOWED_HOSTS=["localhost", "127.0.0.1"]

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Model Paths
MODEL_PATH=./models
MODEL_CACHE_PATH=./models/cache
MODEL_CONFIG_PATH=./config/models

# Model Performance
MAX_BATCH_SIZE=128
GPU_MEMORY_FRACTION=0.85
MODEL_CACHE_TTL=3600
MODEL_RELOAD_INTERVAL=1800

# Inference Settings
DEFAULT_CONFIDENCE_THRESHOLD=0.7
MIN_PREDICTION_CONFIDENCE=0.5
MAX_PREDICTIONS_PER_MINUTE=200
INFERENCE_TIMEOUT=30

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Training Parameters
TRAINING_BATCH_SIZE=64
TRAINING_EPOCHS=100
TRAINING_LEARNING_RATE=0.001
TRAINING_VALIDATION_SPLIT=0.2
TRAINING_TEST_SPLIT=0.1

# Data Processing
DATA_PROCESSING_WORKERS=4
DATA_LOADER_BATCH_SIZE=32
DATA_CACHE_SIZE=1000

# =============================================================================
# API CONFIGURATION
# =============================================================================
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
API_LOG_LEVEL=info

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=20

# =============================================================================
# FRONTEND CONFIGURATION
# =============================================================================
FRONTEND_URL=http://localhost:3000
FRONTEND_PORT=3000
FRONTEND_HOST=0.0.0.0

# =============================================================================
# MONITORING & LOGGING
# =============================================================================
# Logging Configuration
LOG_FILE_PATH=./logs
LOG_FILE_NAME=football-ai.log
LOG_MAX_SIZE=10485760
LOG_BACKUP_COUNT=5
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Monitoring
PROMETHEUS_METRICS_PORT=9090
GRAFANA_PORT=3001
JAEGER_ENDPOINT=http://localhost:14268/api/traces

# =============================================================================
# DUTCHING SYSTEM CONFIGURATION
# =============================================================================
# Dutching Parameters
MIN_DUTCHING_PROFIT_MARGIN=0.05
MAX_DUTCHING_STAKE_PERCENTAGE=0.1
DUTCHING_MAX_BOOKMAKERS=5
DUTCHING_UPDATE_INTERVAL=60

# Risk Management
MAX_SINGLE_BET_STAKE=100
MAX_DAILY_STAKE=1000
RISK_TOLERANCE=medium
STOP_LOSS_PERCENTAGE=0.1

# =============================================================================
# BOOKMAKER CONFIGURATION
# =============================================================================
# Bookmaker APIs (Ersetzen mit echten Keys!)
BET365_API_KEY=your_bet365_api_key
BETFAIR_API_KEY=your_betfair_api_key
WILLIAM_HILL_API_KEY=your_william_hill_api_key
LADBROKES_API_KEY=your_ladbrokes_api_key

# =============================================================================
# NOTIFICATION CONFIGURATION
# =============================================================================
# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Discord/Telegram Webhooks
DISCORD_WEBHOOK_URL=your_discord_webhook_url
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# =============================================================================
# DEVELOPMENT OVERRIDES
# =============================================================================
# Override for development (wird in .env.local gesetzt)
# DEBUG=true
# LOG_LEVEL=DEBUG
# ENVIRONMENT=development
EOF

    log "✅ .env Datei erstellt"
}

# Create backend requirements files
create_requirements_files() {
    log "📋 Erstelle requirements.txt Dateien..."
    
    # Main requirements
    cat > ~/football-ai-system/backend/requirements.txt << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
starlette==0.27.0
pydantic==2.5.0
python-multipart==0.0.6
aiofiles==23.2.1

# Database
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
asyncpg==0.29.0
redis==5.0.1
aioredis==2.0.1

# Machine Learning
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
tensorflow==2.14.0
jax[cuda12_pip]==0.4.20
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.25.2
scipy==1.11.4

# ML Libraries
xgboost==1.7.6
lightgbm==3.3.5
catboost==1.2.2
optuna==3.4.0
hyperopt==0.2.7
scikit-optimize==0.9.0

# Hugging Face
transformers==4.35.0
datasets==2.14.6
accelerate==0.24.1
evaluate==0.4.1
tokenizers==0.15.0
huggingface-hub==0.19.4

# Computer Vision
opencv-python==4.8.1.78
pillow==10.1.0
albumentations==1.3.1
timm==0.9.12

# NLP
spacy==3.7.2
nltk==3.8.1
gensim==4.3.2
textblob==0.17.1

# Data Processing
polars==0.19.19
pyarrow==14.0.1
fastparquet==2023.10.1

# Security & Auth
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
PyJWT==2.8.0
cryptography==41.0.7

# HTTP Clients
requests==2.31.0
httpx==0.25.2
aiohttp==3.9.1
websockets==12.0
python-socketio==5.10.0

# Monitoring & Logging
prometheus-client==0.19.0
structlog==23.2.0
python-json-logger==2.0.7
sentry-sdk==1.38.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
toml==0.10.2
click==8.1.7
typer==0.9.0
rich==13.7.0
tqdm==4.66.1
schedule==1.2.0
APScheduler==3.10.4

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0
isort==5.12.0
EOF

    # Development requirements
    cat > ~/football-ai-system/backend/requirements-dev.txt << 'EOF'
# Development Tools
jupyter==1.0.0
jupyterlab==4.0.9
ipython==8.18.1
notebook==7.0.6

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
factory-boy==3.3.0

# Code Quality
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0
isort==5.12.0
bandit==1.7.5
safety==2.3.5

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0
mkdocs==1.5.3
mkdocs-material==9.4.14

# Profiling
memory-profiler==0.61.0
line-profiler==4.1.1
py-spy==0.3.14

# Debugging
pdb++==0.10.3
ipdb==0.13.13
wdb==3.3.0

# Performance Testing
locust==2.18.3
k6==0.1.0
EOF

    log "✅ Requirements Dateien erstellt"
}

# Create Docker configuration
create_docker_config() {
    log "🐳 Erstelle Docker Konfiguration..."
    
    # Main docker-compose.yml
    cat > ~/football-ai-system/docker-compose.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: football-ai-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: football_ai_db
      POSTGRES_USER: football_ai
      POSTGRES_PASSWORD: football_ai_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    networks:
      - football-ai-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U football_ai"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: football-ai-redis
    restart: unless-stopped
    command: redis-server --requirepass redis_password --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - football-ai-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # TimescaleDB for time-series data
  timescale:
    image: timescale/timescaledb:latest-pg15
    container_name: football-ai-timescale
    restart: unless-stopped
    environment:
      POSTGRES_DB: football_ai_timescale
      POSTGRES_USER: football_ai
      POSTGRES_PASSWORD: football_ai_password
    ports:
      - "5433:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
    networks:
      - football-ai-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U football_ai"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: football-ai-backend
    restart: unless-stopped
    environment:
      ENVIRONMENT: development
      DEBUG: true
      DATABASE_URL: postgresql://football_ai:football_ai_password@postgres:5432/football_ai_db
      REDIS_URL: redis://:redis_password@redis:6379/0
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - football-ai-network
    command: >
      sh -c "uvicorn app.main:app 
             --host 0.0.0.0 
             --port 8000 
             --reload 
             --reload-dir /app"

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: football-ai-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      REACT_APP_API_URL: http://localhost:8000
      CHOKIDAR_USEPOLLING: true
    depends_on:
      - backend
    networks:
      - football-ai-network
    command: npm start

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: football-ai-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./config/nginx/sites-enabled:/etc/nginx/sites-enabled
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    networks:
      - football-ai-network

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: football-ai-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - football-ai-network

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: football-ai-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    networks:
      - football-ai-network

  # Database Admin - pgAdmin
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: football-ai-pgadmin
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@football-ai.com
      PGADMIN_DEFAULT_PASSWORD: admin123
    ports:
      - "5050:80"
    depends_on:
      - postgres
    networks:
      - football-ai-network

  # Redis Admin - Redis Commander
  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: football-ai-redis-commander
    restart: unless-stopped
    environment:
      REDIS_HOSTS: local:redis:6379:0:redis_password
    ports:
      - "8081:8081"
    depends_on:
      - redis
    networks:
      - football-ai-network

volumes:
  postgres_data:
  redis_data:
  timescale_data:
  prometheus_data:
  grafana_data:

networks:
  football-ai-network:
    driver: bridge
EOF

    # Development docker-compose override
    cat > ~/football-ai-system/docker-compose.override.yml << 'EOF'
version: '3.8'

# Development overrides
services:
  backend:
    volumes:
      - ./backend:/app
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      ENVIRONMENT: development
      DEBUG: true
      RELOAD: true
    command: >
      sh -c "uvicorn app.main:app 
             --host 0.0.0.0 
             --port 8000 
             --reload 
             --reload-dir /app"

  frontend:
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      CHOKIDAR_USEPOLLING: true
    command: npm start

  # Additional development tools
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: football-ai-pgadmin
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@football-ai.com
      PGADMIN_DEFAULT_PASSWORD: admin123
    ports:
      - "5050:80"
    depends_on:
      - postgres

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: football-ai-redis-commander
    restart: unless-stopped
    environment:
      REDIS_HOSTS: local:redis:6379:0:redis_password
    ports:
      - "8081:8081"
    depends_on:
      - redis
EOF

    log "✅ Docker Konfiguration erstellt"
}

# Create backend main files
create_backend_files() {
    log "💻 Erstelle Backend Grundstruktur..."
    
    # Main application file
    cat > ~/football-ai-system/backend/app/main.py << 'EOF'
"""
Football AI System - Main FastAPI Application
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging
from logging.config import dictConfig

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.api import api_router
from app.core.database import engine, Base

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Football AI System...")
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Football AI System...")

# Create FastAPI app
app = FastAPI(
    title="Football AI System API",
    description="Advanced AI-powered football prediction and dutching system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🏆 Football AI System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2025-01-06T00:00:00Z",
        "version": "1.0.0",
        "services": {
            "api": "running",
            "database": "connected",
            "redis": "connected"
        }
    }

@app.get("/api/v1/status")
async def system_status():
    """System status endpoint"""
    try:
        # Check database connection
        from app.core.database import check_db_connection
        db_status = await check_db_connection()
        
        # Check Redis connection
        from app.core.redis import check_redis_connection
        redis_status = await check_redis_connection()
        
        # Check GPU availability
        import torch
        gpu_available = torch.cuda.is_available()
        
        return {
            "status": "operational",
            "database": db_status,
            "redis": redis_status,
            "gpu": gpu_available,
            "gpu_count": torch.cuda.device_count() if gpu_available else 0,
            "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS if not settings.DEBUG else 1
    )
EOF

    # Core config
    cat > ~/football-ai-system/backend/app/core/config.py << 'EOF'
"""
Application configuration settings
"""

from typing import List, Optional, Union
from pydantic import BaseSettings, validator
import os
from pathlib import Path

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this"
    JWT_SECRET: str = "your-jwt-secret-change-this"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_MINUTES: int = 1440
    
    # Database
    DATABASE_URL: str = "postgresql://football_ai:password@localhost:5432/football_ai_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "football_ai_db"
    POSTGRES_USER: str = "football_ai"
    POSTGRES_PASSWORD: str = "password"
    
    # Redis
    REDIS_URL: str = "redis://:password@localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "password"
    REDIS_DB: int = 0
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Model Configuration
    MODEL_PATH: str = "./models"
    MAX_BATCH_SIZE: int = 128
    GPU_MEMORY_FRACTION: float = 0.85
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    MIN_PREDICTION_CONFIDENCE: float = 0.5
    MAX_PREDICTIONS_PER_MINUTE: int = 200
    
    # API Keys
    SPORTMONKS_API_KEY: str = ""
    FOOTBALL_DATA_API_KEY: str = ""
    ODDS_API_KEY: str = ""
    WEATHER_API_KEY: str = ""
    
    # Logging
    LOG_FILE_PATH: str = "./logs"
    LOG_FILE_NAME: str = "football-ai.log"
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
EOF

    log "✅ Backend Grundstruktur erstellt"
}

# Create frontend package.json
create_frontend_package() {
    log "🎨 Erstelle Frontend package.json..."
    
    cat > ~/football-ai-system/frontend/package.json << 'EOF'
{
  "name": "football-ai-frontend",
  "version": "1.0.0",
  "private": true,
  "description": "Football AI System - React Frontend",
  "dependencies": {
    "@testing-library/jest-dom": "^5.16.5",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "@types/jest": "^27.5.2",
    "@types/node": "^16.18.68",
    "@types/react": "^18.2.42",
    "@types/react-dom": "^18.2.17",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "typescript": "^4.9.5",
    "web-vitals": "^2.1.4",
    "@tanstack/react-query": "^5.8.4",
    "axios": "^1.6.2",
    "react-router-dom": "^6.18.0",
    "@types/react-router-dom": "^5.3.3",
    "zustand": "^4.4.7",
    "@reduxjs/toolkit": "^1.9.7",
    "react-redux": "^8.1.3",
    "react-hook-form": "^7.48.2",
    "@hookform/resolvers": "^3.3.2",
    "yup": "^1.3.3",
    "socket.io-client": "^4.7.4",
    "@types/socket.io-client": "^3.0.0",
    "date-fns": "^2.30.0",
    "@date-io/date-fns": "^2.17.0",
    "lodash": "^4.17.21",
    "@types/lodash": "^4.14.202",
    "classnames": "^2.3.2",
    "react-toastify": "^9.1.3",
    "framer-motion": "^10.16.5",
    "react-spring": "^9.7.3"
  },
  "devDependencies": {
    "tailwindcss": "^3.3.6",
    "@tailwindcss/forms": "^0.5.7",
    "@tailwindcss/typography": "^0.5.10",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "@headlessui/react": "^1.7.17",
    "@heroicons/react": "^2.0.18",
    "styled-components": "^6.1.1",
    "@types/styled-components": "^5.1.32",
    "recharts": "^2.10.1",
    "victory": "^36.6.11",
    "@types/d3": "^7.4.3",
    "d3": "^7.8.5",
    "react-chartjs-2": "^5.2.0",
    "chart.js": "^4.4.0",
    "@fortawesome/fontawesome-free": "^6.4.2",
    "@fortawesome/react-fontawesome": "^0.2.0",
    "@fortawesome/fontawesome-svg-core": "^6.4.2",
    "@fortawesome/free-solid-svg-icons": "^6.4.2",
    "@fortawesome/free-brands-svg-icons": "^6.4.2",
    "eslint": "^8.54.0",
    "@typescript-eslint/parser": "^6.12.0",
    "@typescript-eslint/eslint-plugin": "^6.12.0",
    "prettier": "^3.0.3"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write src/**/*.{ts,tsx,css}",
    "type-check": "tsc --noEmit",
    "dev": "npm start",
    "prod": "npm run build && npm run serve",
    "serve": "serve -s build",
    "analyze": "npm run build && npx source-map-explorer 'build/static/js/*.js'"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://localhost:8000"
}
EOF

    log "✅ Frontend package.json erstellt"
}

# Create utility scripts
create_utility_scripts() {
    log "🛠️ Erstelle Utility Scripts..."
    
    # Development setup script
    cat > ~/football-ai-system/scripts/setup-dev.sh << 'EOF'
#!/bin/bash
# Development Environment Setup Script

set -e

echo "🚀 Setting up Football AI Development Environment..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate football-ai

# Install pre-commit hooks
pre-commit install

# Setup git hooks
cat > .git/hooks/pre-commit << 'EOF2'
#!/bin/bash
# Pre-commit hook

# Run black formatting
black --check .
if [ $? -ne 0 ]; then
    echo "Code formatting failed. Run 'black .' to fix."
    exit 1
fi

# Run flake8
flake8 .
if [ $? -ne 0 ]; then
    echo "Code style check failed."
    exit 1
fi

# Run mypy
mypy .
if [ $? -ne 0 ]; then
    echo "Type checking failed."
    exit 1
fi
EOF2

chmod +x .git/hooks/pre-commit

echo "✅ Development environment setup complete!"
echo "🔧 Available commands:"
echo "  make dev     - Start development server"
echo "  make test    - Run tests"
echo "  make lint    - Run linting"
echo "  make format  - Format code"
EOF

    chmod +x ~/football-ai-system/scripts/setup-dev.sh
    
    # Database initialization script
    cat > ~/football-ai-system/scripts/setup-database.sh << 'EOF'
#!/bin/bash
# Database Initialization Script

set -e

echo "🗄️ Initializing databases..."

# Wait for PostgreSQL
until pg_isready -h localhost -p 5432 -U football_ai; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

# Wait for Redis
until redis-cli -h localhost -p 6379 ping; do
    echo "Waiting for Redis..."
    sleep 2
done

# Run database migrations
echo "Running database migrations..."
cd backend
alembic upgrade head

# Initialize default data
echo "Initializing default data..."
python -c "
from app.core.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash
from sqlalchemy.orm import Session

db: Session = SessionLocal()

# Create default admin user if not exists
admin = db.query(User).filter(User.email == 'admin@football-ai.com').first()
if not admin:
    admin = User(
        email='admin@football-ai.com',
        username='admin',
        hashed_password=get_password_hash('admin123'),
        is_active=True,
        is_superuser=True
    )
    db.add(admin)
    db.commit()
    print('✅ Default admin user created')
else:
    print('✅ Admin user already exists')

db.close()
"

echo "✅ Database initialization complete!"
EOF

    chmod +x ~/football-ai-system/scripts/setup-database.sh
    
    log "✅ Utility Scripts erstellt"
}

# Create Makefile
create_makefile() {
    log "🔨 Erstelle Makefile..."
    
    cat > ~/football-ai-system/Makefile << 'EOF'
# Football AI System Makefile

.PHONY: help install dev test lint format clean docker-build docker-up docker-down

# Default target
help:
	@echo "🏆 Football AI System"
	@echo ""
	@echo "Verfügbare Befehle:"
	@echo "  make install     - Komplettes System installieren"
	@echo "  make dev         - Development Server starten"
	@echo "  make test        - Tests ausführen"
	@echo "  make lint        - Code linting"
	@echo "  make format      - Code formatieren"
	@echo "  make clean       - Bereinigen"
	@echo "  make docker-up   - Docker Container starten"
	@echo "  make docker-down - Docker Container stoppen"
	@echo "  make backup      - System backup erstellen"
	@echo ""

# Installation
install:
	@echo "🚀 Installiere Football AI System..."
	./01-python-setup-full.sh
	./02-project-setup-full.sh
	@echo "✅ Installation abgeschlossen!"

# Development
dev:
	@echo "🔄 Starte Development Server..."
	@source ~/miniconda3/etc/profile.d/conda.sh && conda activate football-ai
	@cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@cd frontend && npm start &
	@wait

# Testing
test:
	@echo "🧪 Führe Tests aus..."
	@cd backend && pytest -v --cov=app
	@cd frontend && npm test

# Code quality
lint:
	@echo "🔍 Führe Linting aus..."
	@cd backend && flake8 app
	@cd backend && mypy app
	@cd frontend && npm run lint

format:
	@echo "✨ Formatiere Code..."
	@cd backend && black app
	@cd backend && isort app
	@cd frontend && npm run format

# Cleanup
clean:
	@echo "🧹 Bereinige System..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@docker system prune -f

# Docker
docker-build:
	@echo "🐳 Erstelle Docker Images..."
	@docker-compose build

docker-up:
	@echo "🚀 Starte Docker Container..."
	@docker-compose up -d

docker-down:
	@echo "🛑 Stoppe Docker Container..."
	@docker-compose down

docker-logs:
	@echo "📋 Zeige Docker Logs..."
	@docker-compose logs -f

# Backup
backup:
	@echo "💾 Erstelle System Backup..."
	@mkdir -p backups
	@tar -czf backups/football-ai-backup-$$(date +%Y%m%d-%H%M%S).tar.gz \
		--exclude=node_modules \
		--exclude=__pycache__ \
		--exclude=.pytest_cache \
		--exclude=logs \
		--exclude=backups \
		.

# Database
db-migrate:
	@echo "🗄️ Führe Datenbank-Migrationen aus..."
	@cd backend && alembic upgrade head

db-reset:
	@echo "🔄 Setze Datenbank zurück..."
	@cd backend && alembic downgrade base && alembic upgrade head

# Models
model-train:
	@echo "🧠 Trainiere Modelle..."
	@cd backend && python -m app.ml.train

model-predict:
	@echo "🔮 Führe Vorhersagen aus..."
	@cd backend && python -m app.ml.predict

# Development tools
dev-install:
	@echo "🔧 Installiere Development Tools..."
	@pip install -r backend/requirements-dev.txt
	@cd frontend && npm install

setup-git:
	@echo "⚙️  Konfiguriere Git Hooks..."
	@./scripts/setup-dev.sh

# Performance
profile:
	@echo "📊 Erstelle Performance-Profil..."
	@cd backend && python -m cProfile -o profile.stats app/main.py

benchmark:
	@echo "⚡ Führe Benchmarks aus..."
	@cd backend && locust -f tests/locustfile.py --host=http://localhost:8000

# Documentation
docs:
	@echo "📚 Generiere Dokumentation..."
	@cd docs && make html

# System status
status:
	@echo "📋 System Status:"
	@docker-compose ps
	@echo ""
	@echo "📊 System Ressourcen:"
	@htop -n 1 | head -20 || top -n 1 | head -20
	@echo ""
	@echo "🐳 Docker Status:"
	@docker system df
EOF

    log "✅ Makefile erstellt"
}

# Main execution
main() {
    check_user
    
    log "🏗️ Starte Project Structure Setup"
    log "📁 Erstelle vollständige Projektstruktur..."
    
    create_project_structure
    create_env_file
    create_requirements_files
    create_docker_config
    create_backend_files
    create_frontend_package
    create_utility_scripts
    create_makefile
    
    log "✅ Project Structure Setup abgeschlossen!"
    log "📝 Nächster Schritt: .env Datei anpassen und API-Keys eintragen"
    log "🚀 Danach: ./03-ml-models.sh ausführen"
}

# Execute main function
main "$@"