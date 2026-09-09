@echo off
setlocal
call "E:\Software\Anaconda\condabin\conda.bat" activate llama2-7b
if errorlevel 1 exit /b 1
call "E:\Software\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
set "CC=cl.exe"
set "CXX=cl.exe"
set "CUDACXX=%CONDA_PREFIX%\Library\bin\nvcc.exe"
set "CUDA_PATH=%CONDA_PREFIX%\Library"
cd /d "%~dp0.."
"%CUDACXX%" --version
if errorlevel 1 exit /b 1
cmake -S llama.cpp -B llama.cpp/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CUDA_COMPILER="%CUDACXX%" -DCUDAToolkit_ROOT="%CUDA_PATH%" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
if errorlevel 1 exit /b 1
if /i "%~1"=="configure" exit /b 0
cmake --build llama.cpp/build-cuda -j 4 --target llama-server llama-completion llama-quantize llama-bench llama-perplexity
exit /b %errorlevel%
