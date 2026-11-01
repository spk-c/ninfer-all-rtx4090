param(
    [string]$Out = '.\profiles\bench\rtx4090-audit\ncu-matched-k5',
    [switch]$ReuseExisting
)
$ErrorActionPreference='Stop'
Set-Location C:\coding\ninfer-all
$auditNcu='C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe'
$auditOut=[IO.Path]::GetFullPath($Out)
New-Item -ItemType Directory -Force $auditOut | Out-Null
$auditStatuses=@()
foreach ($auditType in @(@('IQ4_XS',23),@('IQ3_S',21),@('IQ3_XXS',18))) {
  foreach ($auditTile in @(128,6)) {
    $auditStem=Join-Path $auditOut "$($auditType[0])-tile$auditTile"
    if (Test-Path "$auditStem.ncu-repz") {
        if (-not $ReuseExisting) { throw 'existing NCU report; select a new -Out or explicitly -ReuseExisting' }
        & $auditNcu --import "$auditStem.ncu-repz" --page raw --csv --print-fp *> "$auditStem.csv"
        if ($LASTEXITCODE -ne 0) { throw 'existing NCU raw export failed' }
        Write-Output "Reused existing NCU capture: $($auditType[0]) tile=$auditTile"
        continue
    }
    $auditFilter=if ($auditTile -eq 128) { "regex:mul_mat_q<\(ggml_type\)$($auditType[1]), \(int\)128," } else { "regex:vec::kernel<\(ggml_type\)$($auditType[1]), \(int\)6," }
    $auditArgs=@('--profile-from-start','off','--clock-control','none',
        '--section','LaunchStats','--section','Occupancy','--section','MemoryWorkloadAnalysis','--section','SpeedOfLight',
        '--launch-count','1','--kernel-name-base','demangled','--kernel-name',$auditFilter,'--export',$auditStem,
        (Join-Path $PWD 'build-rtx4090/bench/ninfer_bench.exe'),
        '--weights','C:\ninfer-all\models\Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer',
        '--kv-dtype','rk4v4','--max-ctx','8192','--prefill-chunk','1024','--spec','dflash2',
        '--draft-tokens','5','--ngram-draft-tokens','0','--lm-head-draft','--gdn-state-fp16',
        '-pg','2048,128','-r','1','--warmup','1','-o','json','--output-file',"$auditStem.json",'--profile-measured')
    nvidia-smi *> "$auditStem-gpu-before.txt"
    & $auditNcu @auditArgs *> "$auditStem.log"
    $auditCode=$LASTEXITCODE
    $auditStatus=[ordered]@{family=$auditType[0];tile=$auditTile;command=@($auditNcu)+$auditArgs;returncode=$auditCode}
    $auditStatuses+= $auditStatus
    $auditStatuses | ConvertTo-Json -Depth 8 | Set-Content "$auditOut/status.json" -Encoding utf8
    if ($auditCode -ne 0) { throw "NCU failed: $($auditType[0]); inspect preserved log" }
    if (-not (Test-Path "$auditStem.ncu-repz")) { throw 'NCU captured no kernel report' }
    & $auditNcu --import "$auditStem.ncu-repz" --page raw --csv --print-fp *> "$auditStem.csv"
    if ($LASTEXITCODE -ne 0) { throw 'NCU raw export failed' }
    Write-Output "NCU complete: $($auditType[0]) tile=$auditTile"
  }
}
