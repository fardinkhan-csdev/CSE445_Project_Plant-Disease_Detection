@echo off
cd /d "%~dp0"
title Leaf Disease Classification - One-Click Downloader
py -3.11 download_assets.py
echo.
pause
