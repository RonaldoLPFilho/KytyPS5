@echo off
rem The settings window (launcher.ps1): game folder, resolution, fullscreen, console language, precompile.
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0launcher.ps1"
