@echo off
setlocal
cd /d "C:\ninfer-all\inkling2-converter-466f027e"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "G=atomicchat-adapter\guard-test"
set "FLAGS=/nologo /std:c++20 /EHsc /MD /O2 /Gy /Zc:__cplusplus /Zc:preprocessor /utf-8 /DNOMINMAX /Iinclude /Isrc /Ithird_party"
set "LIBS=build-rank1-sm89\src\artifact\ninfer_artifact.lib build-rank1-sm89\src\models\ninfer_model_loading.lib build-rank1-sm89\src\core\ninfer_core.lib"
cl %FLAGS% /DEXPECT_LEGACY tests\models\inkling2\rank1_config_guard.cpp "%G%\legacy.cpp" /Fo"%G%\\" /Fe"%G%\legacy.exe" /link /OPT:REF %LIBS%
if errorlevel 1 exit /b 1
"%G%\legacy.exe" "%G%\config.json"
if errorlevel 1 exit /b 1
cl %FLAGS% tests\models\inkling2\rank1_config_guard.cpp src\models\inkling2\config.cpp /Fo"%G%\\" /Fe"%G%\guarded.exe" /link /OPT:REF %LIBS%
if errorlevel 1 exit /b 1
"%G%\guarded.exe" "%G%\config.json"
exit /b %errorlevel%
