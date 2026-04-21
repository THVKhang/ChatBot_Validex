@echo off
echo ===================================================
echo Khởi động Server với Virtual Environment (.venv)
echo ===================================================
call .venv\Scripts\activate
python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
