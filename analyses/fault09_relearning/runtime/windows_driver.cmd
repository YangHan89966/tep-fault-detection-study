@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0.." || exit /b 1
set "F9_SELECTED="
set "F9_SELECTED_OPT="
set "F9_BASE="
set "F9_BASE_OPT="
if defined F9_PYTHON call :probe "%F9_PYTHON%"
call :probe "%CD%\.f9-venv-win\Scripts\python.exe"
if defined VIRTUAL_ENV call :probe "%VIRTUAL_ENV%\Scripts\python.exe"
if defined CONDA_PREFIX call :probe "%CONDA_PREFIX%\python.exe"
where py.exe >nul 2>&1 && call :probe "py.exe" "-3.12"
where python.exe >nul 2>&1 && call :probe "python.exe"
if defined F9_SELECTED goto compatible_native
where wsl.exe >nul 2>&1
if errorlevel 1 goto setup_or_fail
wsl.exe --cd "%CD%" --exec bash ./RUN_WSL.sh --probe-runtime >nul 2>&1
if errorlevel 1 goto setup_or_fail
if /i "%F9_LAUNCH_ACTION%"=="setup" goto already_wsl
wsl.exe --cd "%CD%" --exec bash ./RUN_WSL.sh %*
goto finish
:compatible_native
if /i "%F9_LAUNCH_ACTION%"=="setup" goto already_native
"%F9_SELECTED%" %F9_SELECTED_OPT% "%CD%\runtime\launch.py" %*
goto finish
:already_wsl
echo Compatible CPython 3.12 / NumPy 2.3.5 already available in WSL. Start RUN_FAULT9.cmd.
goto success
:already_native
echo Compatible CPython 3.12 / NumPy 2.3.5 already available. Start RUN_FAULT9.cmd.
goto success
:setup_or_fail
if /i not "%F9_LAUNCH_ACTION%"=="setup" goto no_runtime
if defined F9_BASE goto setup_native
where wsl.exe >nul 2>&1
if errorlevel 1 goto no_runtime
wsl.exe --cd "%CD%" --exec bash ./RUN_WSL.sh --setup-runtime
goto finish
:setup_native
"%F9_BASE%" %F9_BASE_OPT% "%CD%\runtime\setup_local.py"
goto finish
:no_runtime
echo No compatible runtime was found. Install CPython 3.12 if absent, then use SETUP_ENV.cmd.
echo An existing Windows interpreter can be specified with the F9_PYTHON environment variable.
set "F9_EXIT=12"
goto close
:probe
if defined F9_SELECTED exit /b 0
if not exist "%~1" if /i not "%~1"=="py.exe" if /i not "%~1"=="python.exe" exit /b 0
"%~1" %~2 -c "import sys;sys.exit(not(sys.implementation.name=='cpython' and sys.version_info[:2]==(3,12)))" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined F9_BASE (
set "F9_BASE=%~1"
set "F9_BASE_OPT=%~2"
)
"%~1" %~2 -c "import numpy,sys;sys.exit(numpy.__version__!='2.3.5')" >nul 2>&1
if errorlevel 1 exit /b 0
set "F9_SELECTED=%~1"
set "F9_SELECTED_OPT=%~2"
exit /b 0
:success
set "F9_EXIT=0"
goto close
:finish
set "F9_EXIT=%ERRORLEVEL%"
:close
popd
if not "%F9_EXIT%"=="0" echo Launcher returned error %F9_EXIT%. Read the messages above.
if not defined F9_NO_PAUSE pause
exit /b %F9_EXIT%
