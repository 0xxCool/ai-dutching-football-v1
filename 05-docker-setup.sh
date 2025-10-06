#!/bin/bash
# Football AI System - Phase 5: Docker Deployment Setup
# Production-ready Docker configuration with monitoring and security

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

# Navigate to project directory
setup_environment() {
    log "🐳 Starte Docker Deployment Setup..."
    cd ~/football-ai-system
}

# Create Dockerfiles for backend and frontend
create_dockerfiles() {
    log "🐳 Erstelle Dockerfiles..."
    
    # Backend Dockerfile
    if [ ! -f "backend/Dockerfile" ]; then
        cat > backend/Dockerfile << 'EOF'
# Multi-stage build for backend
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data \
    && chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
        log "✅ Backend Dockerfile erstellt"
    fi
    
    # Frontend Dockerfile (already created in frontend-setup)
    if [ ! -f "frontend/Dockerfile" ]; then
        log "❌ Frontend Dockerfile fehlt! Führe Frontend-Setup aus."
        exit 1
    fi
}

# Create production docker-compose file
create_production_compose() {
    log "🐳 Erstelle Production Docker Compose..."
    
    cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

# Production configuration
services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: football-ai-postgres-prod
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-football_ai_db}
      POSTGRES_USER: ${POSTGRES_USER:-football_ai}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-football_ai_password}
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgres/init-scripts:/docker-entrypoint-initdb.d
      - ./backups/postgres:/backups
    networks:
      - football-ai-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-football_ai}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: football-ai-redis-prod
    restart: unless-stopped
    command: >
      redis-server 
      --requirepass ${REDIS_PASSWORD:-redis_password}
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --maxmemory-samples 5
    volumes:
      - redis_data:/data
      - ./config/redis/redis.conf:/usr/local/etc/redis/redis.conf
    networks:
      - football-ai-network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  # TimescaleDB for time-series data
  timescale:
    image: timescale/timescaledb:latest-pg15
    container_name: football-ai-timescale-prod
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${TIMESCALE_DB:-football_ai_timescale}
      POSTGRES_USER: ${TIMESCALE_USER:-football_ai}
      POSTGRES_PASSWORD: ${TIMESCALE_PASSWORD:-football_ai_password}
    volumes:
      - timescale_data:/var/lib/postgresql/data
      - ./config/timescale/init-scripts:/docker-entrypoint-initdb.d
    networks:
      - football-ai-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${TIMESCALE_USER:-football_ai}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
      args:
        - BUILD_ENV=production
    container_name: football-ai-backend-prod
    restart: unless-stopped
    environment:
      ENVIRONMENT: production
      DEBUG: false
      LOG_LEVEL: INFO
      DATABASE_URL: postgresql://${POSTGRES_USER:-football_ai}:${POSTGRES_PASSWORD:-football_ai_password}@postgres:5432/${POSTGRES_DB:-football_ai_db}
      REDIS_URL: redis://:${REDIS_PASSWORD:-redis_password}@redis:6379/0
      WORKERS: 4
      GPU_MEMORY_FRACTION: 0.8
      MAX_BATCH_SIZE: 64
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
      - ./config/backend:/app/config
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      timescale:
        condition: service_healthy
    networks:
      - football-ai-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 4G
          cpus: '2.0'

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - REACT_APP_API_URL=http://backend:8000
        - GENERATE_SOURCEMAP=false
    container_name: football-ai-frontend-prod
    restart: unless-stopped
    depends_on:
      - backend
    networks:
      - football-ai-network
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 256M
          cpus: '0.25'

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    container_name: football-ai-nginx-prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.prod.conf:/etc/nginx/nginx.conf
      - ./config/nginx/sites-enabled:/etc/nginx/sites-enabled
      - ./ssl:/etc/nginx/ssl
      - ./config/nginx/snippets:/etc/nginx/snippets
    depends_on:
      - backend
      - frontend
    networks:
      - football-ai-network
    deploy:
      resources:
        limits:
          memory: 128M
          cpus: '0.1'

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: football-ai-prometheus-prod
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.prod.yml:/etc/prometheus/prometheus.yml
      - ./config/prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - football-ai-network
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: football-ai-grafana-prod
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin123}
      GF_SECURITY_SECRET_KEY: ${GRAFANA_SECRET_KEY:-grafana_secret_key}
      GF_USERS_ALLOW_SIGN_UP: false
      GF_INSTALL_PLUGINS: grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - ./config/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    networks:
      - football-ai-network
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.25'

  # Log Management - Fluentd
  fluentd:
    image: fluent/fluentd:v1.16-debian-1
    container_name: football-ai-fluentd-prod
    restart: unless-stopped
    volumes:
      - ./config/fluentd/fluent.conf:/fluentd/etc/fluent.conf
      - ./logs:/var/log/football-ai
    networks:
      - football-ai-network
    depends_on:
      - elasticsearch
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.25'

  # Log Storage - Elasticsearch
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: football-ai-elasticsearch-prod
    restart: unless-stopped
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - football-ai-network
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Log Visualization - Kibana
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: football-ai-kibana-prod
    restart: unless-stopped
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    networks:
      - football-ai-network
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  # Alert Manager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: football-ai-alertmanager-prod
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./config/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - ./config/alertmanager/templates:/etc/alertmanager/templates
    networks:
      - football-ai-network
    deploy:
      resources:
        limits:
          memory: 128M
          cpus: '0.1'

volumes:
  postgres_data:
  redis_data:
  timescale_data:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
  elasticsearch_data:

networks:
  football-ai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

    log "✅ Production Docker Compose erstellt"
}

# Create monitoring configuration
create_monitoring_config() {
    log "📊 Erstelle Monitoring Konfiguration..."
    
    mkdir -p ~/football-ai-system/config/prometheus
    mkdir -p ~/football-ai-system/config/grafana/provisioning/{dashboards,datasources}
    mkdir -p ~/football-ai-system/config/alertmanager
    mkdir -p ~/football-ai-system/config/fluentd
    
    # Prometheus configuration
    cat > ~/football-ai-system/config/prometheus/prometheus.prod.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'football-ai'
    environment: 'production'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'football-ai-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'football-ai-frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'elasticsearch'
    static_configs:
      - targets: ['elasticsearch:9200']
    metrics_path: '/_prometheus/metrics'
EOF

    # Prometheus rules
    mkdir -p ~/football-ai-system/config/prometheus/rules
    cat > ~/football-ai-system/config/prometheus/rules/football-ai.yml << 'EOF'
groups:
  - name: football-ai-system
    rules:
      # High-level system alerts
      - alert: FootballAISystemDown
        expr: up{job=~"football-ai-.*"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Football AI System is down"
          description: "{{ $labels.job }} has been down for more than 2 minutes."

      # API performance alerts
      - alert: HighAPIResponseTime
        expr: http_request_duration_seconds{job="football-ai-backend", quantile="0.95"} > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API response time"
          description: "95th percentile response time is {{ $value }}s for {{ $labels.job }}."

      # Error rate alerts
      - alert: HighErrorRate
        expr: rate(http_requests_total{job="football-ai-backend", status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for {{ $labels.job }}."

      # Memory usage alerts
      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes{job="football-ai-backend"} / 1024 / 1024) > 2048
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB for {{ $labels.job }}."

      # GPU memory alerts
      - alert: HighGPUMemoryUsage
        expr: (nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes) > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High GPU memory usage"
          description: "GPU memory usage is {{ $value | humanizePercentage }} on {{ $labels.gpu }}."

      # Database alerts
      - alert: PostgreSQLDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL has been down for more than 1 minute."

      - alert: PostgreSQLHighConnections
        expr: pg_stat_activity_count > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "PostgreSQL high connection count"
          description: "PostgreSQL has {{ $value }} active connections."

      # Redis alerts
      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis has been down for more than 1 minute."

      - alert: RedisHighMemoryUsage
        expr: (redis_memory_used_bytes / redis_memory_max_bytes) > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Redis high memory usage"
          description: "Redis memory usage is {{ $value | humanizePercentage }}."

      # Prediction accuracy alerts
      - alert: LowPredictionAccuracy
        expr: football_ai_prediction_accuracy < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low prediction accuracy"
          description: "Prediction accuracy is {{ $value | humanizePercentage }}."

      # Model inference time alerts
      - alert: HighInferenceTime
        expr: football_ai_inference_duration_seconds > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High model inference time"
          description: "Model inference time is {{ $value }}s."
EOF

    # Grafana datasource configuration
    cat > ~/football-ai-system/config/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      httpMethod: POST
      queryTimeout: 60s
      timeInterval: 15s
EOF

    # Grafana dashboard configuration
    cat > ~/football-ai-system/config/grafana/provisioning/dashboards/football-ai.yml << 'EOF'
apiVersion: 1
providers:
  - name: 'Football AI Dashboards'
    orgId: 1
    folder: 'Football AI'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    # Alertmanager configuration
    cat > ~/football-ai-system/config/alertmanager/alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@football-ai.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'admin@football-ai.com'
        subject: 'Football AI Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Labels: {{ .Labels }}
          {{ end }}
    webhook_configs:
      - url: 'http://localhost:5001/webhook'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
EOF

    # Fluentd configuration
    cat > ~/football-ai-system/config/fluentd/fluent.conf << 'EOF'
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<source>
  @type tail
  path /var/log/football-ai/*.log
  pos_file /var/log/fluentd-football-ai.log.pos
  tag football.ai.logs
  <parse>
    @type json
    time_key timestamp
    time_format %Y-%m-%d %H:%M:%S
  </parse>
</source>

<filter football.ai.logs>
  @type grep
  <regexp>
    key level
    pattern ^(ERROR|WARN|INFO)$
  </regexp>
</filter>

<match football.ai.logs>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name football-ai-logs
  type_name _doc
  <buffer>
    flush_interval 5s
  </buffer>
</match>
EOF

    log "✅ Monitoring Konfiguration erstellt"
}

# Create security configuration
create_security_config() {
    log "🔒 Erstelle Security Konfiguration..."
    
    mkdir -p ~/football-ai-system/config/nginx/{sites-enabled,snippets}
    
    # Nginx security configuration
    cat > ~/football-ai-system/config/nginx/nginx.prod.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    # Basic settings
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        application/atom+xml
        application/javascript
        application/json
        application/rss+xml
        application/vnd.ms-fontobject
        application/x-font-ttf
        application/x-web-app-manifest+json
        application/xhtml+xml
        application/xml
        font/opentype
        image/svg+xml
        image/x-icon
        text/css
        text/plain
        text/x-component;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Include site configurations
    include /etc/nginx/sites-enabled/*;
}
EOF

    # Main site configuration
    cat > ~/football-ai-system/config/nginx/sites-enabled/football-ai.conf << 'EOF'
# Main load balancer configuration
upstream backend {
    least_conn;
    server backend:8000 max_fails=3 fail_timeout=30s;
}

upstream frontend {
    least_conn;
    server frontend:3000 max_fails=3 fail_timeout=30s;
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name _;
    return 301 https://$server_name$request_uri;
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name localhost;
    
    # SSL configuration
    ssl_certificate /etc/nginx/ssl/football-ai.crt;
    ssl_certificate_key /etc/nginx/ssl/football-ai.key;
    ssl_dhparam /etc/nginx/ssl/dhparam.pem;
    
    # Security
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # API
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # WebSocket
    location /ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
    
    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header X-Content-Type-Options "nosniff";
    }
    
    # Security.txt
    location /.well-known/security.txt {
        return 200 "Contact: security@football-ai.com\nExpires: 2025-12-31T23:59:59.000Z\n";
        add_header Content-Type text/plain;
    }
}
EOF

    # Security snippets
    cat > ~/football-ai-system/config/nginx/snippets/security.conf << 'EOF'
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

# Hide nginx version
server_tokens off;

# Prevent access to sensitive files
location ~ /\. {
    deny all;
    access_log off;
    log_not_found off;
}

location ~* \.(sql|conf|ini|log|sh|yml|yaml)$ {
    deny all;
    access_log off;
    log_not_found off;
}
EOF

    log "✅ Security Konfiguration erstellt"
}

# Create deployment scripts
create_deployment_scripts() {
    log "🚀 Erstelle Deployment Scripts..."
    
    # Main deployment script
    cat > ~/football-ai-system/scripts/deploy.sh << 'EOF'
#!/bin/bash
# Football AI System Deployment Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
        error "Please run as footballai user (use: su - footballai)"
        exit 1
    fi
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT=${1:-development}
    SKIP_BUILD=${2:-false}
    
    case $ENVIRONMENT in
        development|dev)
            COMPOSE_FILE="docker-compose.yml"
            ;;
        production|prod)
            COMPOSE_FILE="docker-compose.prod.yml"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            echo "Usage: $0 [development|production] [skip-build]"
            exit 1
            ;;
    esac
    
    log "🚀 Deploying to $ENVIRONMENT environment"
    log "📋 Using compose file: $COMPOSE_FILE"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running. Please start Docker daemon."
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        error "docker-compose is not installed. Please install docker-compose."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        warn ".env file not found. Creating from template..."
        cp .env.example .env 2>/dev/null || true
    fi
    
    # Check if required directories exist
    mkdir -p logs backups config/nginx/sites-enabled config/nginx/snippets
    
    log "✅ Pre-deployment checks passed"
}

# Build Docker images
build_images() {
    if [ "$SKIP_BUILD" = "true" ]; then
        log "⚡ Skipping Docker build"
        return
    fi
    
    log "🔨 Building Docker images..."
    
    # Build backend
    if [ -f "backend/Dockerfile" ]; then
        log "Building backend image..."
        docker-compose -f $COMPOSE_FILE build backend
    fi
    
    # Build frontend
    if [ -f "frontend/Dockerfile" ]; then
        log "Building frontend image..."
        docker-compose -f $COMPOSE_FILE build frontend
    fi
    
    log "✅ Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "🚀 Deploying services..."
    
    # Pull latest images
    log "📦 Pulling latest images..."
    docker-compose -f $COMPOSE_FILE pull
    
    # Start services
    log "🚀 Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    # Wait for services to be healthy
    log "⏳ Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log "✅ Services deployed successfully"
}

# Check service health
check_service_health() {
    log "🔍 Checking service health..."
    
    # Check if containers are running
    RUNNING_CONTAINERS=$(docker-compose -f $COMPOSE_FILE ps -q | wc -l)
    TOTAL_CONTAINERS=$(docker-compose -f $COMPOSE_FILE config --services | wc -l)
    
    log "📊 $RUNNING_CONTAINERS/$TOTAL_CONTAINERS containers running"
    
    # Check specific services
    for service in backend frontend postgres redis; do
        if docker-compose -f $COMPOSE_FILE ps $service | grep -q "Up"; then
            log "✅ $service is running"
        else
            warn "⚠️ $service is not running properly"
        fi
    done
    
    # Check API health if backend is running
    if docker-compose -f $COMPOSE_FILE ps backend | grep -q "Up"; then
        sleep 10
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log "✅ API health check passed"
        else
            warn "⚠️ API health check failed"
        fi
    fi
}

# Post-deployment tasks
post_deployment_tasks() {
    log "🔧 Running post-deployment tasks..."
    
    # Run database migrations if in production
    if [ "$ENVIRONMENT" = "production" ]; then
        log "🗄️ Running database migrations..."
        docker-compose -f $COMPOSE_FILE exec -T backend alembic upgrade head || warn "Migration failed"
    fi
    
    # Create admin user if in production
    if [ "$ENVIRONMENT" = "production" ]; then
        log "👤 Creating admin user..."
        docker-compose -f $COMPOSE_FILE exec -T backend python -c "
from app.core.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash
from sqlalchemy.orm import Session

db: Session = SessionLocal()
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
    print('Admin user created')
else:
    print('Admin user already exists')
db.close()
" || warn "Admin user creation failed"
    fi
    
    log "✅ Post-deployment tasks completed"
}

# Show deployment summary
show_summary() {
    log "📋 Deployment Summary"
    echo "===================="
    
    # Show running containers
    docker-compose -f $COMPOSE_FILE ps
    
    echo ""
    echo "🌐 Application URLs:"
    echo "   Frontend: http://localhost:3000"
    echo "   API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "   Grafana: http://localhost:3001 (admin/admin123)"
        echo "   Prometheus: http://localhost:9090"
        echo "   Kibana: http://localhost:5601"
    fi
    
    echo ""
    echo "🔧 Useful Commands:"
    echo "   View logs: docker-compose -f $COMPOSE_FILE logs -f [service]"
    echo "   Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "   Restart service: docker-compose -f $COMPOSE_FILE restart [service]"
    echo "   Execute command: docker-compose -f $COMPOSE_FILE exec [service] [command]"
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (optional)
    # docker volume prune -f
    
    log "✅ Cleanup completed"
}

# Main deployment function
main() {
    check_user
    parse_args "$@"
    pre_deployment_checks
    build_images
    deploy_services
    post_deployment_tasks
    show_summary
    cleanup
    
    log "🎉 Deployment completed successfully!"
    log "🚀 Your Football AI System is now running in $ENVIRONMENT mode"
}

# Execute main function
main "$@"
EOF

    chmod +x ~/football-ai-system/scripts/deploy.sh
    
    # Backup script
    cat > ~/football-ai-system/scripts/backup.sh << 'EOF'
#!/bin/bash
# Football AI System Backup Script

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

BACKUP_DIR="/home/footballai/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

create_backup() {
    local backup_type=$1
    local backup_path="$BACKUP_DIR/${backup_type}_$DATE"
    
    log "Creating $backup_type backup..."
    
    case $backup_type in
        database)
            # PostgreSQL backup
            docker exec football-ai-postgres pg_dump -U football_ai football_ai_db > "${backup_path}_postgres.sql"
            
            # TimescaleDB backup
            docker exec football-ai-timescale pg_dump -U football_ai football_ai_timescale > "${backup_path}_timescale.sql"
            ;;
        
        models)
            # Model files backup
            tar -czf "${backup_path}_models.tar.gz" -C /home/footballai/football-ai-system/models .
            ;;
        
        config)
            # Configuration backup
            tar -czf "${backup_path}_config.tar.gz" -C /home/footballai/football-ai-system/config .
            ;;
        
        full)
            # Full system backup
            docker exec football-ai-postgres pg_dump -U football_ai football_ai_db > "${backup_path}_postgres.sql"
            docker exec football-ai-timescale pg_dump -U football_ai football_ai_timescale > "${backup_path}_timescale.sql"
            tar -czf "${backup_path}_full.tar.gz" -C /home/footballai/football-ai-system \
                --exclude=node_modules \
                --exclude=__pycache__ \
                --exclude=.pytest_cache \
                --exclude=logs \
                --exclude=backups \
                .
            ;;
        
        *)
            error "Unknown backup type: $backup_type"
            exit 1
            ;;
    esac
    
    log "✅ $backup_type backup created: ${backup_path}"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Keep last 7 days of daily backups
    find "$BACKUP_DIR" -name "*_$(date -d '7 days ago' +%Y%m%d)*" -type f -delete
    
    # Keep last 4 weeks of weekly backups
    find "$BACKUP_DIR" -name "weekly_*" -mtime +28 -type f -delete
    
    log "✅ Old backups cleaned up"
}

# Main backup function
main() {
    backup_type=${1:-full}
    
    log "Starting backup process..."
    log "Backup type: $backup_type"
    log "Backup directory: $BACKUP_DIR"
    
    create_backup "$backup_type"
    cleanup_old_backups
    
    log "✅ Backup process completed successfully!"
}

main "$@"
EOF

    chmod +x ~/football-ai-system/scripts/backup.sh
    
    # Monitoring script
    cat > ~/football-ai-system/scripts/monitor.sh << 'EOF'
#!/bin/bash
# Football AI System Monitoring Script

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# System status
show_system_status() {
    log "=== System Status ==="
    
    # Docker containers
    echo "🐳 Docker Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    
    # System resources
    echo "💻 System Resources:"
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "Memory Usage: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2 }')"
    echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"
    
    echo ""
    
    # GPU status
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU Status:"
        nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi
}

# Service health checks
check_service_health() {
    log "=== Service Health Checks ==="
    
    # API health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log "✅ API is healthy"
    else
        error "❌ API is not responding"
    fi
    
    # Database connectivity
    if docker exec football-ai-postgres pg_isready -U football_ai >/dev/null 2>&1; then
        log "✅ PostgreSQL is ready"
    else
        error "❌ PostgreSQL is not ready"
    fi
    
    # Redis connectivity
    if docker exec football-ai-redis redis-cli ping >/dev/null 2>&1; then
        log "✅ Redis is ready"
    else
        error "❌ Redis is not ready"
    fi
    
    # Check disk space
    DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 80 ]; then
        warn "⚠️ Disk usage is high: ${DISK_USAGE}%"
    else
        log "✅ Disk usage is normal: ${DISK_USAGE}%"
    fi
}

# Performance metrics
show_performance_metrics() {
    log "=== Performance Metrics ==="
    
    # API response time
    if command -v curl &> /dev/null; then
        RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" http://localhost:8000/health)
        info "API Response Time: ${RESPONSE_TIME}s"
    fi
    
    # Model inference time (if available)
    if docker exec football-ai-backend python -c "import requests; print(requests.get('http://localhost:8000/api/v1/status').json().get('gpu_available', False))" 2>/dev/null; then
        info "GPU: Available"
    else
        info "GPU: Not available"
    fi
}

# Log analysis
analyze_logs() {
    log "=== Log Analysis ==="
    
    # Recent errors
    ERROR_COUNT=$(docker logs football-ai-backend --since=1h 2>&1 | grep -i error | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        warn "⚠️ Found $ERROR_COUNT errors in the last hour"
    else
        log "✅ No errors found in the last hour"
    fi
    
    # Recent warnings
    WARNING_COUNT=$(docker logs football-ai-backend --since=1h 2>&1 | grep -i warning | wc -l)
    if [ "$WARNING_COUNT" -gt 0 ]; then
        info "ℹ️ Found $WARNING_COUNT warnings in the last hour"
    else
        log "✅ No warnings found in the last hour"
    fi
}

# Network connectivity
check_network_connectivity() {
    log "=== Network Connectivity ==="
    
    # Check external API connectivity
    if curl -f -s https://api.football-data.org/v4/competitions >/dev/null 2>&1; then
        log "✅ External API connectivity is good"
    else
        warn "⚠️ External API connectivity issues detected"
    fi
    
    # Check DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        log "✅ DNS resolution is working"
    else
        error "❌ DNS resolution is not working"
    fi
}

# Generate report
generate_report() {
    log "=== System Health Report ==="
    
    REPORT_FILE="/home/footballai/reports/system_health_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p /home/footballai/reports
    
    {
        echo "Football AI System Health Report"
        echo "Generated: $(date)"
        echo "================================"
        echo ""
        
        echo "Docker Containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        
        echo "System Resources:"
        free -h
        echo ""
        df -h
        echo ""
        
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            nvidia-smi
            echo ""
        fi
        
        echo "Recent Errors:"
        docker logs football-ai-backend --since=1h 2>&1 | grep -i error | tail -10
        echo ""
        
    } > "$REPORT_FILE"
    
    log "📄 Report generated: $REPORT_FILE"
}

# Main monitoring function
main() {
    log "🏥 Starting system health monitoring..."
    
    case "${1:-status}" in
        status)
            show_system_status
            check_service_health
            show_performance_metrics
            ;;
        logs)
            analyze_logs
            ;;
        network)
            check_network_connectivity
            ;;
        report)
            generate_report
            ;;
        full)
            show_system_status
            check_service_health
            show_performance_metrics
            analyze_logs
            check_network_connectivity
            generate_report
            ;;
        *)
            echo "Usage: $0 [status|logs|network|report|full]"
            exit 1
            ;;
    esac
    
    log "✅ Monitoring completed!"
}

main "$@"
EOF

    chmod +x ~/football-ai-system/scripts/monitor.sh
    
    log "✅ Deployment Scripts erstellt"
}

# Create SSL certificate generation script
create_ssl_script() {
    log "🔒 Erstelle SSL Certificate Script..."
    
    cat > ~/football-ai-system/scripts/generate-ssl.sh << 'EOF'
#!/bin/bash
# SSL Certificate Generation Script for Football AI System

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Create SSL directory
SSL_DIR="/home/footballai/football-ai-system/ssl"
mkdir -p "$SSL_DIR"

cd "$SSL_DIR"

# Generate self-signed certificate for development
generate_self_signed() {
    log "Generating self-signed certificate for development..."
    
    # Generate private key
    openssl genrsa -out football-ai.key 4096
    
    # Generate certificate
    openssl req -new -x509 -key football-ai.key -out football-ai.crt -days 365 \
        -subj "/C=DE/ST=Berlin/L=Berlin/O=Football AI/CN=localhost" \
        -config <(cat <<EOF
[req]
default_bits = 4096
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = DE
ST = Berlin
L = Berlin
O = Football AI
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
    )
    
    # Generate DH parameters
    openssl dhparam -out dhparam.pem 2048
    
    log "✅ Self-signed certificate generated"
}

# Generate Let's Encrypt certificate (for production)
generate_letsencrypt() {
    local domain=${1:-localhost}
    
    log "Generating Let's Encrypt certificate for $domain..."
    
    # Install certbot if not available
    if ! command -v certbot &> /dev/null; then
        log "Installing certbot..."
        sudo apt update
        sudo apt install -y certbot
    fi
    
    # Generate certificate
    sudo certbot certonly --standalone \
        -d $domain \
        --email admin@football-ai.com \
        --agree-tos \
        --no-eff-email \
        --keep-until-expiring
    
    # Copy certificates to SSL directory
    sudo cp /etc/letsencrypt/live/$domain/fullchain.pem football-ai.crt
    sudo cp /etc/letsencrypt/live/$domain/privkey.pem football-ai.key
    sudo chown footballai:footballai football-ai.crt football-ai.key
    
    # Generate DH parameters
    openssl dhparam -out dhparam.pem 2048
    
    log "✅ Let's Encrypt certificate generated"
}

# Verify certificate
verify_certificate() {
    log "Verifying certificate..."
    
    if openssl x509 -in football-ai.crt -text -noout >/dev/null 2>&1; then
        log "✅ Certificate is valid"
    else
        error "❌ Certificate is invalid"
        exit 1
    fi
}

# Main function
main() {
    local cert_type=${1:-self-signed}
    local domain=${2:-localhost}
    
    log "Starting SSL certificate generation..."
    log "Certificate type: $cert_type"
    log "Domain: $domain"
    
    case $cert_type in
        self-signed|dev|development)
            generate_self_signed
            ;;
        letsencrypt|prod|production)
            generate_letsencrypt "$domain"
            ;;
        *)
            error "Invalid certificate type: $cert_type"
            echo "Usage: $0 [self-signed|letsencrypt] [domain]"
            exit 1
            ;;
    esac
    
    verify_certificate
    
    log "✅ SSL certificate generation completed!"
    log "📍 Certificates location: $SSL_DIR"
}

main "$@"
EOF

    chmod +x ~/football-ai-system/scripts/generate-ssl.sh
    
    log "✅ SSL Certificate Script erstellt"
}

# Main execution
main() {
    check_user
    setup_environment
    
    log "🚀 Starte Docker Deployment Setup"
    
    # Create configurations
    create_dockerfiles
    create_production_compose
    create_monitoring_config
    create_security_config
    create_deployment_scripts
    create_ssl_script
    
    # Create additional directories
    mkdir -p ~/football-ai-system/backups/postgres
    mkdir -p ~/football-ai-system/config/postgres/init-scripts
    mkdir -p ~/football-ai-system/config/redis
    mkdir -p ~/football-ai-system/config/timescale/init-scripts
    mkdir -p ~/football-ai-system/config/backend
    mkdir -p ~/football-ai-system/reports
    
    # Create init scripts
    touch ~/football-ai-system/config/postgres/init-scripts/init.sql
    touch ~/football-ai-system/config/redis/redis.conf
    touch ~/football-ai-system/config/timescale/init-scripts/init.sql
    touch ~/football-ai-system/config/backend/config.yml
    
    log "✅ Docker Deployment Setup abgeschlossen!"
    log "🚀 Deployment starten: ./scripts/deploy.sh production"
    log "📊 Monitoring: ./scripts/monitor.sh"
    log "💾 Backup: ./scripts/backup.sh"
    log "🔒 SSL Certificate: ./scripts/generate-ssl.sh"
}

# Execute main function
main "$@"