@echo off
cd /d "%~dp0"
echo Starting Metric Studio with Viewer...
start http://localhost:8002/static/index.html
python -m uvicorn api_main:app --host 127.0.0.1 --port 8002
pause
