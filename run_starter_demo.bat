@echo off
setlocal
cd /d "%~dp0"

set "PORT="
for /l %%P in (8501,1,8510) do (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort %%P -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>nul
  if not defined PORT if not errorlevel 1 set "PORT=%%P"
)

if not defined PORT (
  echo No free Streamlit port found between 8501 and 8510.
  pause
  exit /b 1
)

echo Starting NZSL Recognition Demo...
echo Open http://localhost:%PORT%
".venv\Scripts\python.exe" -m streamlit run starter_nzsl/app/streamlit_app.py --server.address localhost --server.port %PORT%

if errorlevel 1 (
  echo.
  echo Streamlit exited with an error.
  pause
)
