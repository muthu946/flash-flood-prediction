@echo off
title Push AI Flash Flood Prediction to GitHub (muthu946)
color 0b
echo ========================================================
echo   Pushing AI Flash Flood Project to GitHub: muthu946
echo   Target: https://github.com/muthu946/flash-flood-prediction
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/5] Initializing local Git repository...
git init
git config user.name "muthu946"
git config user.email "muthu946@users.noreply.github.com"

echo [2/5] Staging all project files and datasets...
git add .

echo [3/5] Creating initial commit...
git commit -m "feat: AI Flash Flood Early Warning System with 100% SIL-4 Safety Interlock"

echo [4/5] Setting primary branch to 'main'...
git branch -M main

echo [5/5] Configuring remote repository origin...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/muthu946/flash-flood-prediction.git

echo.
echo ========================================================
echo   Uploading files to GitHub (git push)...
echo ========================================================
git push -u origin main --force

echo.
echo ========================================================
echo   Upload Complete! View your project online at:
echo   https://github.com/muthu946/flash-flood-prediction
echo ========================================================
echo.
pause
