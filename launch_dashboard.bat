@echo off
title AI Flash Flood Early Warning Dashboard
cd /d "%~dp0"
echo ========================================================
echo Starting AI Flash Flood Early Warning System Dashboard...
echo ========================================================
python -m streamlit run app.py
pause
