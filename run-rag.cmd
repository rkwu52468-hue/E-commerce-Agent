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
set "SHOP_RAG_PYTHON=%~dp0.venv-rag\Scripts\python.exe"
if not exist "%SHOP_RAG_PYTHON%" set "SHOP_RAG_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SHOP_RAG_PYTHON%" (
    echo Create .venv-rag or .venv and install requirements-rag.txt before starting.
    exit /b 1
)
if /I "%~1"=="live" goto live
pushd "%~dp0"
"%SHOP_RAG_PYTHON%" -B -m customer_service.knowledge_cli %*
set "SHOP_RAG_EXIT=%ERRORLEVEL%"
popd
exit /b %SHOP_RAG_EXIT%

:live
if not "%~2"=="" (
    echo Additional live flags: use python app.py --live --allow-knowledge [flags].
    exit /b 2
)
pushd "%~dp0"
"%SHOP_RAG_PYTHON%" -B "%~dp0app.py" --live --allow-knowledge
set "SHOP_RAG_EXIT=%ERRORLEVEL%"
popd
exit /b %SHOP_RAG_EXIT%
