@echo off
setlocal

cd /d "%~dp0"

echo Starting MIoT IDS realtime API at http://127.0.0.1:8000
echo Keep this terminal open while using the React UI.
echo.

.venv\Scripts\python.exe scripts\ids_realtime_api.py ^
  --host 127.0.0.1 ^
  --port 8000 ^
  --run-dir reports\full_gpu_hpo_models_20260306_195851 ^
  --test-csv data\merged\metadata_test.csv ^
  --rows-per-second 1 ^
  --local-top-n 8 ^
  --replay-order shuffle ^
  --shuffle-seed 42 ^
  --max-recent-alerts 250

endlocal
