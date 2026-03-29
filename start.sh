#!/bin/bash
set -e
cd /home/ubuntu/Mlflow
source venv/bin/activate

# ── UPDATE THIS EVERY TIME THE INSTANCE RESTARTS ──────────────────────────────
SERVER_IP="13.232.63.176"
# ──────────────────────────────────────────────────────────────────────────────

echo "==> Starting SentimentScope..."

# 1. Nginx (frontend)
sudo systemctl start nginx 2>/dev/null || true
echo "[1/3] Nginx     → http://$SERVER_IP/app"

# 2. MLflow server
if ! ss -tlnp | grep -q ':5001'; then
    nohup mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlartifacts \
        --host 127.0.0.1 \
        --port 5001 \
        > mlflow_server.log 2>&1 &
    sleep 4
fi
echo "[2/3] MLflow    → http://$SERVER_IP/mlflow/"

# 3. Flask API
pkill -f "python flask_app/app.py" 2>/dev/null || true
sleep 1
nohup python flask_app/app.py > flask_app.log 2>&1 &
sleep 6

echo "[3/3] Flask API → http://$SERVER_IP/"

# Health checks
echo ""
echo "==> Health checks..."
ss -tlnp | grep -q ':80'   && echo "  [✓] Nginx   (port 80)"   || echo "  [✗] Nginx   FAILED"
ss -tlnp | grep -q ':5001' && echo "  [✓] MLflow  (port 5001)" || echo "  [✗] MLflow  FAILED"
ss -tlnp | grep -q ':5000' && echo "  [✓] Flask   (port 5000)" || echo "  [✗] Flask   FAILED"

echo ""
echo "All services up!"
echo "  Frontend : http://$SERVER_IP/app"
echo "  Flask API: http://$SERVER_IP/"
echo "  MLflow   : http://$SERVER_IP/mlflow/"
