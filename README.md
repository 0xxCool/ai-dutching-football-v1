# 🚀 Server-Implementierungsanleitung - Football AI System

## Übersicht

Diese Anleitung beschreibt die vollständige Implementierung des Football AI Bet Dutching Systems auf einem Ubuntu 24.04 Server mit RTX 3090 GPU.

## 📋 Voraussetzungen

### Hardware
- **GPU**: NVIDIA RTX 3090 24GB VRAM ✅
- **CPU**: Min. 8 vCPUs
- **RAM**: 32GB+ RAM
- **Storage**: 500GB+ SSD
- **Network**: 1Gbps Verbindung

### Software (bereits vorhanden)
- **OS**: Ubuntu 24.04 LTS ✅
- **CUDA**: Version 12.7 ✅
- **NVIDIA Treiber**: 565.57.01 ✅
- **cuDNN**: Version 9.13 ✅

## 🛠️ Schritt-für-Schritt Implementierung

### Phase 1: System-Setup

```bash
# 1. System aktualisieren
sudo apt update && sudo apt upgrade -y

# 2. Benutzer erstellen
sudo useradd -m -s /bin/bash footballai
sudo usermod -aG sudo footballai
su - footballai

# 3. Miniconda installieren
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 4. Conda Environment erstellen
conda create -n football-ai python=3.10 -y
conda activate football-ai
```

### Phase 2: Python Environment

```bash
# Alle Skripte ausführen
chmod +x 01-python-setup.sh
./01-python-setup.sh
```

**Enthält:**
- PyTorch 2.1.0 mit CUDA 12.7 Support
- TensorFlow 2.14.0 GPU Version
- JAX mit GPU Support
- Hugging Face Transformers
- FastAPI & Web Frameworks
- ML Libraries (Scikit-learn, XGBoost, LightGBM)

### Phase 3: Projekt-Struktur

```bash
# Projektverzeichnis erstellen
./02-project-setup.sh
```

**Erstellte Struktur:**
```
~/football-ai-system/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   └── schemas/
│   ├── models/
│   ├── services/
│   └── utils/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── services/
│   └── public/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── config/
├── scripts/
└── .env
```

### Phase 4: ML-Modelle

```bash
# ML-Modelle einrichten
./03-ml-models.sh
```

**Features:**
- Base Model Classes
- Neural Networks für verschiedene Prediction Types
- Ensemble Methoden
- GPU-optimierte Inference
- Model Registry

### Phase 5: Frontend-Setup

```bash
# React Frontend installieren
./04-frontend-setup.sh
```

**Technologien:**
- React 18 mit TypeScript
- Tailwind CSS
- Real-time WebSocket Updates
- Chart.js & Recharts
- Responsive Design

### Phase 6: Docker-Deployment

```bash
# Docker-Setup
./05-docker-setup.sh

# Services starten
docker-compose up -d
```

## 🔧 Konfiguration

### .env Konfiguration

```env
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://football_ai:secure_password@postgres:5432/football_ai_db
REDIS_URL=redis://:secure_password@redis:6379

# API Keys (ersetzen mit echten Keys)
SPORTMONKS_API_KEY=your_api_key_here
ODDS_API_KEY=your_api_key_here
FOOTBALL_DATA_API_KEY=your_api_key_here
WEATHER_API_KEY=your_api_key_here

# Security
SECRET_KEY=your_secure_secret_key
JWT_SECRET=your_secure_jwt_secret

# Model Configuration
MODEL_PATH=./models
MAX_BATCH_SIZE=128
GPU_MEMORY_FRACTION=0.85

# Performance
DEFAULT_CONFIDENCE_THRESHOLD=0.7
MAX_PREDICTIONS_PER_MINUTE=200
```

### Datenbank-Setup

```bash
# PostgreSQL initialisieren
docker exec -it postgres psql -U football_ai -d football_ai_db

# Tabellen erstellen
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    match_id INTEGER,
    prediction_type VARCHAR(50),
    predicted_outcome VARCHAR(100),
    confidence FLOAT,
    odds DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    home_team VARCHAR(100),
    away_team VARCHAR(100),
    match_date TIMESTAMP,
    league VARCHAR(100),
    status VARCHAR(20)
);
```

## 🚀 Produktions-Deployment

### SSL-Zertifikat

```bash
# Let's Encrypt installieren
sudo apt install certbot python3-certbot-nginx -y

# Zertifikat erstellen
sudo certbot --nginx -d deinedomain.com -d www.deinedomain.com

# Auto-Renewal aktivieren
sudo systemctl enable certbot.timer
```

### Nginx Konfiguration

```nginx
# /etc/nginx/sites-available/football-ai
server {
    listen 443 ssl http2;
    server_name deinedomain.com;

    ssl_certificate /etc/letsencrypt/live/deinedomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/deinedomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# HTTP Redirect
server {
    listen 80;
    server_name deinedomain.com;
    return 301 https://$server_name$request_uri;
}
```

### Firewall-Konfiguration

```bash
# UFW Firewall einrichten
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# IPTables Backup
sudo iptables-save > /etc/iptables/rules.v4
```

## 📊 Monitoring-Setup

### Prometheus & Grafana

```bash
# Monitoring Stack starten
docker-compose -f docker-compose.monitoring.yml up -d

# Grafana: http://localhost:3001
# Standard-Zugang: admin/admin
```

### Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: football-ai-alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          
      - alert: LowPredictionAccuracy
        expr: prediction_accuracy < 70
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped significantly"
```

## 🔒 Sicherheit

### Security-Checkliste

- [ ] API-Keys in .env gespeichert
- [ ] Datenbank-Passwörter geändert
- [ ] SSL-Zertifikat installiert
- [ ] Firewall konfiguriert
- [ ] Fail2ban installiert
- [ ] SSH-Key-Only Authentication
- [ ] Automatische Security-Updates

```bash
# Fail2ban installieren
sudo apt install fail2ban -y

# SSH-Hardening
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PubkeyAuthentication yes
sudo systemctl restart sshd

# Automatische Updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

## 🔄 Backup-Strategie

### Automatisierte Backups

```bash
#!/bin/bash
# backup-script.sh

# Datenbank-Backup
docker exec postgres pg_dump -U football_ai football_ai_db > /backup/football_ai_$(date +%Y%m%d).sql

# Modelle sichern
tar -czf /backup/models_$(date +%Y%m%d).tar.gz ~/football-ai-system/models/

# Konfiguration sichern
tar -czf /backup/config_$(date +%Y%m%d).tar.gz ~/football-ai-system/config/ ~/football-ai-system/.env

# Alte Backups aufräumen
find /backup/ -name "*.sql" -mtime +7 -delete
find /backup/ -name "*.tar.gz" -mtime +7 -delete
```

### Cron-Job einrichten

```bash
# Backup täglich um 2 Uhr
0 2 * * * /home/footballai/backup-script.sh

# Log-Rotation
sudo nano /etc/logrotate.d/football-ai
```

## 🚀 Performance-Optimierung

### GPU-Optimierung

```bash
# CUDA Memory Management
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# TensorRT Optimierung
pip install tensorrt
pip install torch2trt
```

### System-Optimierung

```bash
# Kernel-Parameter optimieren
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'net.core.somaxconn=65535' >> /etc/sysctl.conf
sysctl -p

# File-Descriptor Limits
echo 'fs.file-max = 65535' >> /etc/sysctl.conf
echo 'footballai soft nofile 65535' >> /etc/security/limits.conf
echo 'footballai hard nofile 65535' >> /etc/security/limits.conf
```

## 📈 Skalierungs-Strategien

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  backend:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

### Load Balancing

```nginx
# Nginx Load Balancer
upstream backend {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=2;
    server backend3:8000 weight=1;
}
```

## 🐛 Troubleshooting

### Log-Analyse

```bash
# Docker-Logs anzeigen
docker-compose logs -f backend
docker-compose logs -f frontend

# System-Logs
tail -f /var/log/syslog
tail -f /var/log/nginx/error.log

# GPU-Status überprüfen
nvidia-smi -l 1
watch -n 1 nvidia-smi
```

### Performance-Probleme

```bash
# CPU-Auslastung
top -p $(pgrep python)

# Memory-Analyse
free -h
cat /proc/meminfo

# Disk-I/O
iostat -x 1
iotop
```

## ✅ Deployment-Checkliste

### Pre-Deployment

- [ ] Alle Skripte erfolgreich ausgeführt
- [ ] API-Keys konfiguriert
- [ ] Datenbank initialisiert
- [ ] SSL-Zertifikat bereit
- [ ] Domain konfiguriert

### Post-Deployment

- [ ] Alle Services laufen
- [ ] HTTPS-Zugriff funktioniert
- [ ] API-Endpoints erreichbar
- [ ] Monitoring aktiv
- [ ] Backups konfiguriert
- [ ] Alerts funktionieren

### Testing

```bash
# Health-Check
curl -f http://localhost:8000/health
curl -f https://deinedomain.com/api/health

# Load-Testing
ab -n 1000 -c 10 https://deinedomain.com/api/predictions

# GPU-Testing
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎯 Erfolgsmetriken

### KPIs überwachen

- **Inference Latenz**: <15ms
- **Durchsatz**: >100 req/s
- **Genauigkeit**: >85%
- **Verfügbarkeit**: >99.9%
- **Fehler-Rate**: <0.1%

### Reporting

```python
# Beispiel Monitoring-Report
def generate_performance_report():
    metrics = {
        'avg_latency': calculate_avg_latency(),
        'throughput': calculate_throughput(),
        'accuracy': calculate_model_accuracy(),
        'uptime': calculate_system_uptime(),
        'error_rate': calculate_error_rate()
    }
    
    return metrics
```

## 📞 Support & Wartung

### Wartungsplan

- **Täglich**: Logs prüfen, Backups verifizieren
- **Wöchentlich**: System-Updates, Performance-Analyse
- **Monatlich**: Security-Scans, Model-Retraining
- **Quartalsweise**: Hardware-Prüfung, Kapazitätsplanung

### Notfall-Prozeduren

```bash
# Schnell-Restart
docker-compose restart

# Rollback
./rollback.sh v1.2.3

# Disaster Recovery
./disaster-recovery.sh
```

---

**Hinweis**: Diese Anleitung wurde speziell für die TensorDock-Umgebung mit RTX 3090 optimiert. Die vorinstallierten CUDA-Treiber und die Ubuntu 24.04-Konfiguration beschleunigen die Installation erheblich.

**Support**: Bei Fragen oder Problemen wenden Sie sich bitte an das Entwicklungsteam oder konsultieren Sie die Troubleshooting-Sektion.

**Version**: 1.0.0 | **Letzte Aktualisierung**: 2025-01-06
