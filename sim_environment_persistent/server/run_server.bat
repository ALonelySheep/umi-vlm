@echo off
echo Starting simplified Isaac Sim persistent environment with Flask...
echo.
echo This will launch the environment and keep it running until you close it.
echo Press Ctrl+C to stop the server.
echo.

cd %~dp0
cd ../..

REM Install Flask if not already installed
echo Checking if Flask is installed...
pip show flask > nul 2>&1
if %errorlevel% neq 0 (
  echo Installing Flask...
  pip install flask
)

python.bat sim_environment_persistent/server/run_server.py