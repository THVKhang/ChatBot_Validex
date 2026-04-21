@echo off
echo ===================================================
echo Chạy Unit Tests với Virtual Environment (.venv)
echo ===================================================
call .venv\Scripts\activate
python -m pytest -q
