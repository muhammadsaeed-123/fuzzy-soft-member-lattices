@echo off
cd /d "%~dp0"
python code\reproduce.py %*
exit /b %errorlevel%
