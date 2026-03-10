# #!/bin/bash
# set -e
# cd /home/ubuntu/Mlflow
# source venv/bin/activate

# echo "==> Starting SentimentScope..."

# # 1. Nginx (frontend)
# sudo systemctl start nginx 2>/dev/null || true
# echo "[1/3] Nginx     → http://3.111.31.244/app"

# # 2. MLflow server
# if ! ss -tlnp | grep -q ':5001'; then
#     nohup mlflow server \
#         --backend-store-uri sqlite:///mlflow.db \
#         --default-artifact-root ./mlartifacts \
#         --host 127.0.0.1 \
#         --port 5001 \
#         > mlflow_server.log 2>&1 &
#     sleep 3
# fi
# echo "[2/3] MLflow    → http://3.111.31.244/mlflow/"

# # 3. Flask API
# pkill -f "python flask_app/app.py" 2>/dev/null || true
# sleep 1
# nohup python flask_app/app.py > flask_app.log 2>&1 &
# sleep 4

# sleep 2
# echo "[3/3] Flask API → http://3.111.31.244/"

# # Health checks
# echo ""
# echo "==> Health checks..."
# ss -tlnp | grep -q ':80'   && echo "  [✓] Nginx   (port 80)"   || echo "  [✗] Nginx   FAILED"
# ss -tlnp | grep -q ':5001' && echo "  [✓] MLflow  (port 5001)" || echo "  [✗] MLflow  FAILED"
# ss -tlnp | grep -q ':5000' && echo "  [✓] Flask   (port 5000)" || echo "  [✗] Flask   FAILED"

# echo ""
# echo "✓ All services up!"
# echo "  Frontend : http://3.111.31.244/app"
# echo "  Flask API: http://3.111.31.244/"
# echo "  MLflow   : http://3.111.31.244/mlflow/"



#!/bin/bash
set -e

# 🔥 CHANGE THIS ONLY WHEN IP CHANGES
SERVER_IP="13.204.81.139"

echo "==> Starting SentimentScope..."

# 1️⃣ Start Nginx
sudo systemctl start nginx
echo "[1/2] Nginx     → http://$SERVER_IP/app"

# 2️⃣ Start Flask (Docker)

# Remove old container if exists
docker rm -f flask 2>/dev/null || true

# Run Flask container
docker run -d \
--name flask \
-p 5000:5000 \
sentiment-app

sleep 3

echo "[2/2] Flask API → http://$SERVER_IP/"

# ✅ Health checks
echo ""
echo "==> Health checks..."

ss -tlnp | grep -q ':80'   && echo "  [✓] Nginx   (port 80)"   || echo "  [✗] Nginx FAILED"
ss -tlnp | grep -q ':5000' && echo "  [✓] Flask   (port 5000)" || echo "  [✗] Flask FAILED"

echo ""
echo "✅ App is live!"
echo "  Frontend : http://$SERVER_IP/app"
echo "  Backend  : http://$SERVER_IP/"