@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run starter_nzsl/app/streamlit_app.py --server.address localhost --server.port 8501
