@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo Git was not found in PATH.
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
  echo Initializing Git repository...
  git init
)

set "commit_message=update"
set /p "commit_message=Commit message (default: update): "
if "!commit_message!"=="" set "commit_message=update"

git add -A

git status --short

git commit -m "!commit_message!"
if errorlevel 1 (
  echo Commit failed or there was nothing to commit.
  pause
  exit /b 1
)

git rev-parse --abbrev-ref --symbolic-full-name @{u} >nul 2>nul
if errorlevel 1 (
  echo No upstream branch is configured yet.
  echo Add a remote and push manually:
  echo git remote add origin ^<your-repo-url^>
  echo git push -u origin HEAD
) else (
  git push
)
pause
