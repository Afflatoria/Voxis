@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python baseline\cli_speak.py %*
