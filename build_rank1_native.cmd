@echo off
setlocal
cd /d "C:\ninfer-all\inkling2-converter-466f027e"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cmake -S . -B build-rank1-sm89 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe" "-DNINFER_MEDIA_SOURCES_DIR=C:/coding/ninfer-codex-deps/media" "-DNINFER_VCPKG_TREE=C:/coding/ninfer-codex-deps/vcpkg/x64-windows" -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF -DNINFER_SHARED_RUNTIME=OFF "-DPython3_EXECUTABLE=C:/ninfer-all/.venv-convert/Scripts/python.exe" > atomicchat-adapter\native-configure.log 2>&1
if errorlevel 1 exit /b 1
ninja -C build-rank1-sm89 -t targets all > atomicchat-adapter\native-targets.log
if errorlevel 1 exit /b 1
cmake --build build-rank1-sm89 --target ninfer-serve ninfer --parallel 10 > atomicchat-adapter\native-build.log 2>&1
exit /b %errorlevel%
