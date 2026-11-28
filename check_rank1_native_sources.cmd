@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
C:\ninfer-all\.venv-convert\Scripts\python.exe -u "C:\ninfer-all\inkling2-converter-466f027e\check_rank1_native_sources.py"
exit /b %errorlevel%
