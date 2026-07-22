@echo off
setlocal EnableExtensions

cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo Git was not found in PATH.
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
  echo This folder is not a Git repository yet.
  echo Run push.bat once to initialize it.
  pause
  exit /b 1
)

echo Checking for large tracked artifacts that should not be pushed...
for /f "delims=" %%F in ('git ls-files 2^>nul') do (
  set "file=%%~fF"
  if exist "!file!" (
    for %%I in ("!file!") do (
      if %%~zI GTR 52428800 (
        echo WARNING: Large tracked file detected: %%F (%%~zI bytes)
      )
    )
  )
)

git pull --ff-only
if errorlevel 1 (
  echo Pull failed. Check your remote branch or merge conflicts.
  pause
  exit /b 1
)
pause
