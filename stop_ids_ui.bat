@echo off
setlocal

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
  taskkill /PID %%p /F >nul 2>&1
  echo Stopped IDS UI process PID %%p
  goto :done
)

echo No IDS UI process found on port 8501.

:done
endlocal
