# MIoT IDS Prototype (Realtime)

This UI now consumes a local realtime Python API that runs the 3 protocol-routed XGBoost models live over `metadata_test.csv`.

## 1) Start realtime API (required)

From repo root:

```powershell
.venv\Scripts\python.exe scripts/ids_realtime_api.py `
  --host 127.0.0.1 `
  --port 8000 `
  --run-dir reports/full_gpu_hpo_models_20260306_195851 `
  --test-csv data/merged/metadata_test.csv `
  --rows-per-second 1 `
  --local-top-n 8 `
  --replay-order shuffle `
  --shuffle-seed 42 `
  --max-recent-alerts 250
```

Or run:

```powershell
start_ids_realtime_api.bat
```

## 2) Run React app

Open a terminal in `web/ids-react-ui` and run:

```powershell
$env:Path = "C:\Program Files\nodejs;$env:Path"
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open:

`http://127.0.0.1:5173`
