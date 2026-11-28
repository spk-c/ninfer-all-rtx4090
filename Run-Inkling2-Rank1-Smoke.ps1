[CmdletBinding()]
param([ValidateRange(1,65536)][int]$Context=8192)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$exe=Join-Path $root 'build-rank1-sm89\apps\ninfer.exe'
$model=Join-Path (Split-Path -Parent $root) 'models\Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer'
$log=Join-Path $root 'atomicchat-adapter\rank1-smoke.log'
foreach ($file in @($exe,$model)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file" }
}
if (Get-Process -Name ninfer-serve,ninfer -ErrorAction SilentlyContinue) {
    throw 'Another NInfer process is running. Stop it explicitly before this test.'
}
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;'+$env:PATH
Write-Host 'Experimental fixed rank-one model. No speculative decoder; no CUDA graph.'
$ErrorActionPreference='Continue'

& $exe $model --max-context $Context --kv-dtype rk8v4 --expert-residency device --spec none --no-cuda-graph --greedy --max-new 128 --prompt 'Write a Python function that adds two integers and a unit test for it.' 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "NInfer exited $LASTEXITCODE" }
Write-Host 'Process exited successfully; review output and compare numerical reference before qualification.'
