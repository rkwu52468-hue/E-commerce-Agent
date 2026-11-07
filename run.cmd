@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
if not defined PYTHONIOENCODING set "PYTHONIOENCODING=utf-8"
if not defined FASTEMBED_CACHE_PATH set "FASTEMBED_CACHE_PATH=%~dp0.cache\fastembed"
if not defined HF_HOME set "HF_HOME=%~dp0.cache\huggingface"
if not defined HF_HUB_DISABLE_SYMLINKS_WARNING set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not defined TEMP (
    set "TEMP=%~dp0.tmp"
    if not exist "%~dp0.tmp" mkdir "%~dp0.tmp"
)
if not defined TMP set "TMP=%TEMP%"
set "SHOP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SHOP_PYTHON%" set "SHOP_PYTHON=%~dp0.venv-rag\Scripts\python.exe"
if not exist "%SHOP_PYTHON%" (
    echo Create .venv or .venv-rag and install requirements before starting.
    exit /b 1
)
pushd "%~dp0"
"%SHOP_PYTHON%" -B "%~dp0app.py" %*
set "SHOP_EXIT=%ERRORLEVEL%"
popd
exit /b %SHOP_EXIT%
