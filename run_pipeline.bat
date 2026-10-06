@echo off
title Hybrid Athlete Metabolic Analytics Engine
echo ============================================================
echo   Ejecutando Hybrid Athlete & Sports Nutrition Pipeline...
echo ============================================================
echo.
cd /d "%~dp0"
.\.venv\Scripts\python.exe main.py
echo.
echo Presione cualquier tecla para cerrar...
pause >nul
