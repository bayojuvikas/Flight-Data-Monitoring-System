@echo off
REM ===============================================
REM  Aero AI Platform - Project Structure Creator
REM  Usage: double-click this file OR run in CMD
REM ===============================================

SET PROJECT_ROOT=aero_ai_platform

echo Creating project root: %PROJECT_ROOT%
mkdir %PROJECT_ROOT%
cd %PROJECT_ROOT%

REM ---- Top-level files ----
echo Creating top-level files...
type nul > README.md
type nul > requirements.txt

REM ---- .vscode folder ----
echo Creating .vscode folder...
mkdir .vscode
type nul > .vscode\launch.json
type nul > .vscode\tasks.json

REM ---- Python package root ----
echo Creating Python package: aero_ai_platform
mkdir aero_ai_platform
type nul > aero_ai_platform\__init__.py
type nul > aero_ai_platform\config.py

REM ---- data_generation module ----
echo Creating data_generation module...
mkdir aero_ai_platform\data_generation
type nul > aero_ai_platform\data_generation\__init__.py
type nul > aero_ai_platform\data_generation\flight.py
type nul > aero_ai_platform\data_generation\engine.py
type nul > aero_ai_platform\data_generation\shm.py

REM ---- features module ----
echo Creating features module...
mkdir aero_ai_platform\features
type nul > aero_ai_platform\features\__init__.py
type nul > aero_ai_platform\features\flight_features.py
type nul > aero_ai_platform\features\engine_features.py
type nul > aero_ai_platform\features\shm_features.py

REM ---- models module ----
echo Creating models module...
mkdir aero_ai_platform\models
type nul > aero_ai_platform\models\__init__.py
type nul > aero_ai_platform\models\flight.py
type nul > aero_ai_platform\models\engine.py
type nul > aero_ai_platform\models\shm.py

REM ---- ui module ----
echo Creating ui module...
mkdir aero_ai_platform\ui
type nul > aero_ai_platform\ui\__init__.py
type nul > aero_ai_platform\ui\dashboard_streamlit.py

REM ---- scripts folder ----
echo Creating scripts folder...
mkdir scripts
type nul > scripts\generate_synthetic_data.py
type nul > scripts\train_all_models.py

REM ---- data + models_artifacts folders ----
echo Creating data and models_artifacts folders...
mkdir data
mkdir data\raw
mkdir models_artifacts

echo.
echo ===============================================
echo Project structure created under %CD%
echo Now:
echo   1) Open this folder in VS Code
echo   2) Paste the Python code and VS Code JSON
echo      from our ChatGPT conversation into:
echo        - config.py, *.py, launch.json, tasks.json
echo   3) Run:  pip install -r requirements.txt
echo   4) Then run the scripts / Streamlit dashboard
echo ===============================================
echo Done.
pause