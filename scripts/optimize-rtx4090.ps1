# RTX 4090 real-text qualification: no production or GPU clock changes.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Cli = '.\build-rtx4090\apps\Release\ninfer.exe',
    [string]$Out = '',
    [ValidateRange(3,20)][int]$Rounds = 3,
    [ValidateRange(0,7)][int]$Device = 0,
    [ValidateSet('draft','prefill','kv','copy')][string[]]$Groups =
        @('draft','prefill','kv','copy'),
    [switch]$DryRun
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Out) { $Out = Join-Path $Root ('profiles\bench\campaign-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
$Python = Get-Command python -ErrorAction Stop
$Runner = Join-Path $Root 'tools\bench\run_rtx4090_optimization_campaign.py'
if (-not $DryRun) {
    $Model = (Resolve-Path -LiteralPath $Model -ErrorAction Stop).Path
    $Cli = (Resolve-Path -LiteralPath $Cli -ErrorAction Stop).Path
}
$Args = @($Runner, '--model', $Model, '--cli', $Cli,
    '--out', $Out, '--rounds', [string]$Rounds, '--device', [string]$Device,
    '--groups') + @($Groups)
if ($DryRun) { $Args += '--dry-run' }
Push-Location $Root
try {
    & $Python.Source @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Campaign failed; inspect per-case output in $Out"
    }
} finally { Pop-Location }
Write-Host "RTX 4090 campaign complete: $Out"
