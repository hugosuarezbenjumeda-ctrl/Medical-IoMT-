@echo off
setlocal

cd /d "%~dp0"

echo Generating SMALL simulation bundle for UI testing...
.venv\Scripts\python.exe scripts\generate_ids_react_simulation_data.py ^
  --run-dir reports\full_gpu_hpo_models_20260306_195851 ^
  --test-csv data\merged\metadata_test.csv ^
  --output-json web\ids-react-ui\public\simulation_data.json ^
  --window-rows 50 ^
  --window-seconds 50 ^
  --max-alerts-per-window 3 ^
  --max-local-features 8 ^
  --max-windows 5000 ^
  --replay-order interleave-protocol-source ^
  --sampling-strategy balanced-protocol ^
  --shuffle-seed 42

echo Done.
endlocal
