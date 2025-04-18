@echo off
echo Running Isaac Sim Flask client command...
echo.

cd %~dp0
cd ../..

REM Install requests if not already installed
echo Checking if requests is installed...
pip show requests > nul 2>&1
if %errorlevel% neq 0 (
  echo Installing requests...
  pip install requests
)

python.bat sim_environment_persistent/client/run_client.py %*