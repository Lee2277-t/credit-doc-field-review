@echo off
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PY=python"
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -3.11"
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist ".venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"

%PY% -c "import pymupdf,cv2,yaml,numpy,pdfplumber,rapidocr_onnxruntime,sklearn,PIL" >nul 2>&1
if errorlevel 1 (
  echo Python or dependencies are missing. Run these commands in this folder:
  echo   py -3.11 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

%PY% -c "import shutil,pathlib; p=next(pathlib.Path('out/final').glob('*.html'),None); shutil.copy2(p,'.') if p else None"
set "OUT="
if exist "out\summary.json" set "OUT=--out out"
%PY% -u -m docpipe.serve %OUT% --open-browser
if errorlevel 1 pause
