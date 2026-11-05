# Compare separately built upstream NInfer and SPK-C RTX 4090 engines on one quiet GPU.
# Builds stay in their own apps directories; do NOT copy their EXEs into another build tree.
# Each arm uses the exact same .ninfer artifact, token settings and prompts.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$UpstreamExe,
    [Parameter(Mandatory = $true)][string]$ForkExe,
    [ValidateRange(2, 30)][int]$Repetitions = 6,
    [ValidateRange(1, 15)][int]$DraftTokens = 5,
    [ValidateSet('dflash2', 'none')][string]$Spec = 'dflash2',
    [string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $RepoRoot 'tools\bench\run_interleaved_ab.py'
foreach ($Path in @($Model, $UpstreamExe, $ForkExe, $Runner)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}
$Model = (Resolve-Path -LiteralPath $Model).Path
$UpstreamExe = (Resolve-Path -LiteralPath $UpstreamExe).Path
$ForkExe = (Resolve-Path -LiteralPath $ForkExe).Path
if ($UpstreamExe -eq $ForkExe) { throw 'Both executables resolve to the same file' }
foreach ($Exe in @($UpstreamExe, $ForkExe)) {
    if ((Split-Path -Leaf (Split-Path -Parent $Exe)).ToLowerInvariant() -ne 'apps') {
        throw "Each engine must be run from its own build apps directory: $Exe"
    }
}
$Python = Get-Command python -ErrorAction Stop
$Smi = Get-Command nvidia-smi -ErrorAction Stop
$Gpu = @(& $Smi.Source '--query-gpu=name,driver_version,power.limit,temperature.gpu' '--format=csv,noheader')
if ($LASTEXITCODE -ne 0 -or -not ($Gpu -match 'RTX 4090')) {
    throw 'The A/B test requires a detected RTX 4090'
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepoRoot ('profiles\bench\rtx4090-upstream-ab-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
if (Test-Path -LiteralPath $OutputDir) { throw "Refusing to overwrite $OutputDir" }
$null = New-Item -ItemType Directory -Path $OutputDir -Force
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$UpstreamVersion = @(& $UpstreamExe '--version' 2>&1)
if ($LASTEXITCODE -ne 0) { throw 'Upstream CLI --version failed; check its own DLLs' }
$ForkVersion = @(& $ForkExe '--version' 2>&1)
if ($LASTEXITCODE -ne 0) { throw 'Fork CLI --version failed; check its own DLLs' }

$Manifest = [ordered]@{
    timestamp_utc = [DateTime]::UtcNow.ToString('o')
    comparison = 'B: SPK-C RTX4090 fork / A: upstream'
    model = $Model
    model_sha256 = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash
    upstream_exe = $UpstreamExe
    upstream_sha256 = (Get-FileHash -LiteralPath $UpstreamExe -Algorithm SHA256).Hash
    upstream_version = $UpstreamVersion
    fork_exe = $ForkExe
    fork_sha256 = (Get-FileHash -LiteralPath $ForkExe -Algorithm SHA256).Hash
    fork_version = $ForkVersion
    gpu = $Gpu
    repetitions = $Repetitions
    spec = $Spec
    draft_tokens = if ($Spec -eq 'dflash2') { $DraftTokens } else { 0 }
    max_context = 8192
    kv_dtype = 'rk4v4'
    max_new = 256
    note = 'Paired median decode speeds; not a measure of response quality or numerical equivalence. Keep GPU idle and inspect output for drift.'
}
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDir 'manifest.json') -Encoding utf8

# The runner alternates AB/BA within repetitions. Both CLIs remain next to
# their own runtime DLLs; the opt-in preflight validates --version on each.
$Prompt = 'Explain CUDA graph launch overhead and GPU memory bandwidth in five sentences.'
$BaseArgs = '{exe} "' + $Model + '" --prompt "' + $Prompt + '" --max-new 256 --max-context 8192 --kv-dtype rk4v4 --greedy --gdn-state-fp16'
$Configs = @('plain:' + $BaseArgs)
if ($Spec -eq 'dflash2') {
    $Configs += 'dflash2-k' + $DraftTokens + ':' + $BaseArgs + ' --spec dflash2 --draft-tokens ' + $DraftTokens
}
$Arguments = @($Runner, '--arm-a', $UpstreamExe, '--arm-b', $ForkExe,
               '--allow-separate-dirs', '--reps', [string]$Repetitions)
foreach ($Config in $Configs) { $Arguments += @('--config', $Config) }

$Log = Join-Path $OutputDir 'paired-results.log'
Write-Host "Comparing upstream (A) with SPK-C fork (B) on $($Gpu[0])"
Write-Host "Output: $OutputDir"
Push-Location $RepoRoot
try {
    & $Python.Source @Arguments 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) { throw "A/B test incomplete; inspect $Log" }
} finally {
    Pop-Location
}
Write-Host "Completed comparison: $OutputDir"
