#!/usr/bin/env bash
# Запуск Streamlit-приложения на VM (для жюри). Запускается НА VM:
#   bash ~/shlif/deploy/run_app.sh
# Приложение: http://<IP VM>:8501
set -euo pipefail
cd ~/shlif
mkdir -p logs
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1
nohup ./venv/bin/streamlit run app.py \
    --server.address 0.0.0.0 --server.port 8501 \
    --server.maxUploadSize 1024 --server.headless true \
    > logs/streamlit.log 2>&1 &
sleep 3
tail -5 logs/streamlit.log
echo "OK: http://$(hostname -I | awk '{print $1}'):8501"
