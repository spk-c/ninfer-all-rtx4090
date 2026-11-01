# Three reproducible RTX 4090 measurements: graph VRAM, host/CUDA API intervals,
# and IQ4_XS duration share. Read-only against the model and executables.
# This script NEVER tunes kernels or claims a speedup.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Cli = '.\build-rtx4090\apps\ninfer.exe',
    [string]$CandidateCli = '',
    [string]$Bench = '.\build-rtx4090\bench\ninfer_bench.exe',
    [string]$Nsys = 'nsys',
    [string]$Python = 'python',
    [string]$Out = '.\profiles\bench\rtx4090-audit',
    [ValidateRange(2,25)][int]$GraphRepeats = 5,
    [ValidateRange(1,15)][int]$DraftTokens = 5,
    [ValidateRange(0,63)][int]$NgramDraftTokens = 0,
    [switch]$PlanOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $Repo
try {
    $Model = (Resolve-Path -LiteralPath $Model -ErrorAction Stop).Path
    $Cli = (Resolve-Path -LiteralPath $Cli -ErrorAction Stop).Path
    $Bench = (Resolve-Path -LiteralPath $Bench -ErrorAction Stop).Path
    if ($CandidateCli) { $CandidateCli = (Resolve-Path -LiteralPath $CandidateCli -ErrorAction Stop).Path }
    $Out = [IO.Path]::GetFullPath((Join-Path $Repo $Out))
    & $Python -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ required' }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'nvidia-smi required' }
    $GpuNames = @(& nvidia-smi --query-gpu=name --format=csv,noheader)
    if ($LASTEXITCODE -ne 0 -or -not ($GpuNames | Where-Object { $_ -match 'RTX 4090' })) {
        throw 'Run on the RTX 4090 host (nvidia-smi could not find one)'
    }
    if (-not (Get-Command $Nsys -ErrorAction SilentlyContinue) -and
        -not (Test-Path -LiteralPath $Nsys)) { throw 'Nsight Systems nsys not found; specify -Nsys' }

    $graphOut = Join-Path $Out 'graph-memory-baseline'
    $candidateGraphOut = Join-Path $Out 'graph-memory-candidate'
    $stem = Join-Path $Out 'decode-dflash2-k5'
    $kernCsv = $stem + '_cuda_gpu_kern_sum.csv'
    $apiCsv = $stem + '_cuda_api_sum.csv'
    $json = Join-Path $Out 'iq4-xs-share.json'
    $graph = @('-m','tools.bench.run_graph_allowance_sweep',
        '--cli',$Cli,'--model',$Model,'--spec','dflash2',
        '--draft-tokens',[string]$DraftTokens,'--kv-dtype','rk4v4',
        '--ngram-draft-tokens',[string]$NgramDraftTokens,
        '--prefill-chunk','1024','--gdn-state-fp16','--capacities','8192','32768','65536',
        '--repeats',[string]$GraphRepeats,'--output',$graphOut)
    $benchArgs = @('--weights',$Model,'--kv-dtype','rk4v4',
        '--max-ctx','8192','--prefill-chunk','1024','--spec','dflash2','--draft-tokens',[string]$DraftTokens,
        '--ngram-draft-tokens',[string]$NgramDraftTokens,
        '--lm-head-draft','--gdn-state-fp16','-pg','2048,128','-r','1','--warmup','1',
        '-o','json','--output-file',($stem + '.json'),'--profile-measured')
    $candidateGraph = @($graph)
    $candidateGraph[[array]::IndexOf($candidateGraph,'--cli') + 1] = $CandidateCli
    $candidateGraph[[array]::IndexOf($candidateGraph,'--output') + 1] = $candidateGraphOut
    $comparison = @('-m','tools.bench.compare_graph_allowance_sweeps',
        (Join-Path $graphOut 'graph-allowance-dflash2-rk4v4.json'),
        (Join-Path $candidateGraphOut 'graph-allowance-dflash2-rk4v4.json'),
        '--output',(Join-Path $Out 'graph-memory-comparison.json'))
    # osrt is a Linux collector; Windows Nsight rejects it before launching the target.
    $trace = if ($env:OS -eq 'Windows_NT') { 'cuda,nvtx,wddm' } else { 'cuda,nvtx,osrt' }
    $profile = @('profile','--force-overwrite','true','--output',$stem,
        '--trace',$trace,'--cuda-graph-trace','node',
        '--capture-range','cudaProfilerApi','--capture-range-end','stop',
        $Bench) + $benchArgs
    $stats = @('stats','--report','cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_trace',
        '--format','csv','--output',$stem,($stem + '.nsys-rep'))
    $analyse = @('-m','tools.bench.rtx4090_kernel_share',
        '--input',$kernCsv,'--output',$json,'--require-iq4')

    if ($PlanOnly) {
        Write-Host "Graph sweep : python $($graph -join ' ')"
        if ($CandidateCli) {
            Write-Host "Candidate VRAM sweep: python $($candidateGraph -join ' ')"
            Write-Host "Compare memory sweeps: python $($comparison -join ' ')"
        }
        Write-Host "Nsight trace: $Nsys $($profile -join ' ')"
        Write-Host "Nsight stats: $Nsys $($stats -join ' ')"
        Write-Host "IQ4 analysis: python $($analyse -join ' ')"
        Write-Host 'No benchmarks executed in -PlanOnly mode.'
        return
    }
    if (Test-Path -LiteralPath ($stem + '.nsys-rep')) {
        throw "Existing trace would be overwritten: $stem.nsys-rep; select another -Out"
    }
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    $artifact = Get-Item -LiteralPath $Model
    $meta = [ordered]@{
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        source_revision = (& git rev-parse HEAD)
        source_changes = @(& git status --porcelain --untracked-files=no)
        gpu_names = @($GpuNames)
        gpu_state = @(& nvidia-smi --query-gpu=name,driver_version,pstate,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.gr,clocks.mem --format=csv)
        nsys_version = (& $Nsys --version)
        python_version = (& $Python --version)
        telemetry_utc_offset = [DateTimeOffset]::Now.Offset.ToString()
        model_path = $Model
        model_size_bytes = $artifact.Length
        model_modified_utc = $artifact.LastWriteTimeUtc.ToString('o')
        cli_path = $Cli
        cli_version = (& $Cli --version)
        candidate_cli_path = $CandidateCli
        cli_sha256 = (Get-FileHash -LiteralPath $Cli -Algorithm SHA256).Hash
        bench_path = $Bench
        bench_sha256 = (Get-FileHash -LiteralPath $Bench -Algorithm SHA256).Hash
        graph_repeats = $GraphRepeats
        kv_dtype = 'rk4v4'
        prefill_chunk = 1024
        spec = 'dflash2'
        draft_tokens = $DraftTokens
        ngram_draft_tokens = $NgramDraftTokens
        gdn_state_fp16 = $true
        scope = 'Device-wide graph allocation and Nsight CUDA/host activity; no causal speedup claim'
    }
    $meta | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Out 'manifest.json') -Encoding utf8
    Write-Host 'Phase 3a: recording repeated graph-memory allowances'
    $graphTelemetryArgs = @{
        FilePath = (Get-Command nvidia-smi).Source
        ArgumentList = @('--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.gr,clocks.mem', '--format=csv', '--loop-ms=100')
        RedirectStandardOutput = (Join-Path $Out 'graph-telemetry.csv')
        RedirectStandardError = (Join-Path $Out 'graph-telemetry-errors.log')
        PassThru = $true
    }
    if ($env:OS -eq 'Windows_NT') { $graphTelemetryArgs.WindowStyle = 'Hidden' }
    $graphTelemetryProcess = Start-Process @graphTelemetryArgs
    try {
        & $Python @graph
        $graphExitCode = $LASTEXITCODE
    } finally {
        if (-not $graphTelemetryProcess.HasExited) { Stop-Process -Id $graphTelemetryProcess.Id }
        $graphTelemetryProcess.WaitForExit()
    }
    if ($graphExitCode -ne 0) { throw 'Graph allowance sweep incomplete; inspect graph-memory output' }
    if ($CandidateCli) {
        Write-Host 'Candidate graph-memory sweep (separate samples; GPU drift may confound comparisons)'
        & $Python @candidateGraph
        if ($LASTEXITCODE -ne 0) { throw 'Candidate graph allowance sweep incomplete' }
        & $Python @comparison
        if ($LASTEXITCODE -ne 0) { throw 'Cannot compare graph-memory sweeps' }
    }
    Write-Host 'Phase 3b: capturing CUDA graph nodes, NVTX and CUDA API wall intervals'
    $telemetryArgs = @{
        FilePath = (Get-Command nvidia-smi).Source
        ArgumentList = @('--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.gr,clocks.mem,clocks_event_reasons.active', '--format=csv', '--loop-ms=100')
        RedirectStandardOutput = (Join-Path $Out 'gpu-telemetry.csv')
        RedirectStandardError = (Join-Path $Out 'gpu-telemetry-errors.log')
        PassThru = $true
    }
    if ($env:OS -eq 'Windows_NT') { $telemetryArgs.WindowStyle = 'Hidden' }
    $telemetryProcess = Start-Process @telemetryArgs
    try {
        & $Nsys @profile *> (Join-Path $Out 'nsys-profile.log')
        $profileExitCode = $LASTEXITCODE
    } finally {
        if (-not $telemetryProcess.HasExited) { Stop-Process -Id $telemetryProcess.Id }
        $telemetryProcess.WaitForExit()
    }
    if ($profileExitCode -ne 0) { throw 'Nsight profile failed; inspect nsys-profile.log' }
    & $Nsys @stats *> (Join-Path $Out 'nsys-stats.log')
    if ($LASTEXITCODE -ne 0) { throw 'Nsight stats export failed; inspect nsys-stats.log' }
    if (-not (Test-Path $kernCsv) -or -not (Test-Path $apiCsv)) {
        throw 'Nsight kernel or API CSV is missing; do not infer host or kernel times'
    }
    Write-Host 'Phase 3c: calculating measured IQ4_XS time share'
    & $Python @analyse
    if ($LASTEXITCODE -ne 0) { throw 'IQ4_XS attribution could not be validated' }
    Write-Host "RESULTS: $Out"
    Write-Host "Host-side CUDA API summary: $apiCsv"
    Write-Host 'CUDA API durations are not CPU instruction time; uncaptured graph nodes distort percentages.'
} finally {
    Pop-Location
}
