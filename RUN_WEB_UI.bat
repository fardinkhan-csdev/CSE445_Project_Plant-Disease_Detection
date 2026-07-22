@echo off
title FloraGuard AI Web UI
echo ============================================================
echo Starting FloraGuard AI Web Server on http://localhost:8000
echo ============================================================

start "" http://localhost:8000
python web_app/server.py 8000
pause
