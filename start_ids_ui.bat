@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv at: %cd%\.venv
  echo Create it first, then install dependencies.
  pause
  exit /b 1
)

echo Starting IDS UI server on http://localhost:8501 ...
echo Keep this terminal open while using the app.
echo Then open your browser manually to: http://localhost:8501
echo.

".venv\Scripts\python.exe" -m streamlit run scripts/ids_xgb_interpretability_ui.py --server.headless true --server.address 127.0.0.1 --server.port 8501

echo.
echo IDS UI process exited.
echo If this was unexpected, share the error text shown above.
pause

endlocal
