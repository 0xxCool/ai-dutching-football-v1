# 🚀 TensorDock RTX 3090 – Komplette Installationsanleitung  
**Football AI Bet Dutching System**  
Repository: https://github.com/0xxCool/ai-dutching-football-v1

---

## 📋 System-Check – Vorab-Status auf TensorDock
| Komponente | Status | Bemerkung |
|------------|--------|-----------|
| GPU | ✅ **NVIDIA RTX 3090 24 GB** | Wird voll erkannt |
| CUDA | ✅ **12.7** | Bereits installiert |
| Ubuntu | ✅ **24.04 LTS** | Neuer als geplant |
| NVIDIA Treiber | ✅ **565.57.01** | Aktuell |
| cuDNN | ✅ **9.13** | Vorhanden |

➔ **Keine GPU-/CUDA-Arbeiten nötig** – wir starten direkt bei Schritt 2.

---

## 1️⃣ Zugang & erste Sicherheit
```bash
# Verbinde dich über SSH (Daten stehen in der TensorDock-Mail)
ssh root@DEINE_IP -p 22

# 1. Passwort ändern
passwd

# 2. System updaten
apt update && apt upgrade -y

# 3. Essentials installieren
apt install -y curl wget git nano htop build-essential
```

---

## 2️⃣ Benutzer „footballai“ erstellen (NICHT als root weitermachen!)
```bash
adduser --gecos "" footballai
usermod -aG sudo footballai

# SSH-Key für footballai (optional, aber empfohlen)
mkdir -p /home/footballai/.ssh
chmod 700 /home/footballai/.ssh
# Eigenen Public-Key in /home/footballai/.ssh/authorized_keys einfügen
chown -R footballai:footballai /home/footballai/.ssh
chmod 600 /home/footballai/.ssh/authorized_keys

# Wechsel zu footballai
su - footballai
```

---

## 3️⃣ Repository klonen & Skripte ausführbar machen
```bash
cd ~
git clone https://github.com/0xxCool/ai-dutching-football-v1.git football-ai
cd football-ai
chmod +x *.sh
```

---

## 4️⃣ Miniconda + football-ai Environment
```bash
# Miniconda installieren
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
rm miniconda.sh

# Environment erstellen
conda create -n football-ai python=3.10 -y
conda activate football-ai
```

---

## 5️⃣ Phase-Skripte ausführen (jeweils ca. 5-15 min)
```bash
# 5.1 Python & ML Stack (PyTorch, TF, JAX, FastAPI, …)
./01-python-setup-full.sh

# 5.2 Projektstruktur + .env + Docker-Compose
./02-project-setup-full.sh

# 5.3 ML-Modelle + Registry
./03-ml-models-full.sh

# 5.4 React-Frontend (inkl. npm-Build)
./04-frontend-setup-full.sh

# 5.5 Docker-Images bauen + Monitoring-Stack
./05-docker-setup-full.sh
```

---

## 6️⃣ .env anpassen (API-Keys & Passwörter)
```bash
nano ~/football-ai/.env
```
**Mindestens ändern:**
```env
# Security
SECRET_KEY=EinSehrLangesZufallsstring123!§$
JWT_SECRET=NochEinLangerZufallsstring456!§$

# Datenbank-Passwörter
POSTGRES_PASSWORD=EinSicheresDBPasswort
REDIS_PASSWORD=EinSicheresRedisPasswort

# API-Keys (hier echte Keys eintragen!)
SPORTMONKS_API_KEY=dein_echter_key
ODDS_API_KEY=dein_echter_key
FOOTBALL_DATA_API_KEY=dein_echter_key
```

---

## 7️⃣ SSL-Zertifikat (Let’s Encrypt) – nur für Produktion mit Domain
```bash
sudo apt install certbot python3-certbot-nginx -y
# Beispiel-Domain ersetzen
sudo certbot --nginx -d deinedomain.com -d www.deinedomain.com
# Auto-Renewal testen
sudo certbot renew --dry-run
```

---

## 8️⃣ Production-Deployment starten
```bash
cd ~/football-ai
# Images bauen & Stack starten
docker-compose -f docker-compose.prod.yml up -d --build

# Status prüfen
docker-compose -f docker-compose.prod.yml ps
```

---

## 9️⃣ Erreichbare Endpunkte nach 1-2 Minuten
| Service | URL | Login |
|---------|-----|-------|
| Frontend | `https://deinedomain.com` | – |
| API Docs | `https://deinedomain.com/docs` | – |
| Grafana | `https://deinedomain.com:3001` | `admin / admin123` |
| Prometheus | `https://deinedomain.com:9090` | – |
| Kibana | `https://deinedomain.com:5601` | – |

---

## 🔧 Wartung & häufige Befehle
```bash
# Logs live ansehen
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f frontend

# Neustart einzelner Services
docker-compose -f docker-compose.prod.yml restart backend

# Backup auslösen
./scripts/backup.sh full

# Monitoring-Report
./scripts/monitor.sh full

# Updates ziehen
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --build
```

---

## 🛡️ Security-Hardening (optional aber empfohlen)
```bash
# Fail2ban installieren
sudo apt install fail2ban -y
sudo systemctl enable fail2ban

# SSH nur mit Key
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PubkeyAuthentication yes
sudo systemctl restart sshd

# Automatische Updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## 🎯 Performance-Tuning (RTX 3090)
```bash
# .env – RTX 3090 optimiert
MAX_BATCH_SIZE=128          # 24 GB VRAM
GPU_MEMORY_FRACTION=0.85    # 85 % der GPU
DEFAULT_CONFIDENCE_THRESHOLD=0.7
INFERENCE_TIMEOUT=15        # ms
```

---

## 📊 KPI-Überwachung
- **Inference Latenz**: <15 ms  
- **Durchsatz**: >100 req/s  
- **Genauigkeit**: >85 %  
- **Verfügbarkeit**: >99,9 %  
- **Fehler-Rate**: <0,1 %  

---

## 🚨 Notfall-Restart
```bash
# Kompletter Stack neu starten
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d --build

# Rollback (Beispiel)
./scripts/rollback.sh v1.0.0
```

---

## 📞 Support & nächste Schritte
- **GitHub Issues**: https://github.com/0xxCool/ai-dutching-football-v1/issues  
- **Discord**: https://discord.gg/football-ai  
- **E-Mail**: support@football-ai.com  

---

✅ **Fertig!**  
Dein Football AI Bet Dutching System läuft nun **produktionsbereit** auf TensorDock mit RTX 3090, CUDA 12.7 und Ubuntu 24.04.
