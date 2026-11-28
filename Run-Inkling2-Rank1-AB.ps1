$ErrorActionPreference='Continue'
$root=$PSScriptRoot
$exe=Join-Path $root 'build-rank1-sm89\apps\ninfer.exe'
$models=Join-Path (Split-Path -Parent $root) 'models'
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;'+$env:PATH
if (Get-Process -Name ninfer-serve,ninfer -ErrorAction SilentlyContinue) { throw 'Another NInfer process is running' }
$prompt='Write a Python function that adds two integers and a unit test for it.'
foreach ($case in @(@{Name='original';File='Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer'},@{Name='rank1';File='Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer'})) {
    $model=Join-Path $models $case.File
    $log=Join-Path $root ('atomicchat-adapter\ab-'+$case.Name+'.log')
    Write-Host "CASE $($case.Name)"
    & $exe $model --max-context 8192 --kv-dtype rk8v4 --expert-residency device --spec none --no-cuda-graph --greedy --max-new 512 --prompt $prompt 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "NInfer exited $LASTEXITCODE" }
}
