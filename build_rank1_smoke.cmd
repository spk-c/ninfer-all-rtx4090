@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "C:\ninfer-all\inkling2-converter-466f027e"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" -arch=sm_89 -std=c++17 -O2 tests\models\inkling2\rank1_adapter_smoke.cu -o rank1_adapter_smoke.exe
if errorlevel 1 exit /b 1
rank1_adapter_smoke.exe
exit /b %errorlevel%
