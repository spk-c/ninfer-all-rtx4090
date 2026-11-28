# Run from a normal Windows user session with Docker Desktop already started.
[CmdletBinding()]
param([string]$Image='python:3.11-slim')
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$inputDir=Join-Path $root 'atomicchat-adapter\humaneval'
$stage=Join-Path $inputDir 'evaluation-stage'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($file in @('HumanEval.jsonl.gz','original.jsonl','rank1.jsonl')) {
    Copy-Item -LiteralPath (Join-Path $inputDir $file) -Destination (Join-Path $stage $file) -Force
}
Copy-Item -LiteralPath (Join-Path $root 'evaluate_rank1_humaneval.py') -Destination (Join-Path $stage 'evaluate_rank1_humaneval.py') -Force
& docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop is not available in this session.' }
$kind=& docker info --format '{{.OSType}}'
if ($kind -ne 'linux') { throw 'Linux containers required.' }
& docker image inspect $Image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw "Image not present. Pull explicitly: docker pull $Image" }
$log=Join-Path $inputDir 'functional-results.json'
if (Test-Path -LiteralPath $log) { throw "Result exists: $log" }
$result=& docker run --rm --pull never --network none --read-only --tmpfs '/tmp:rw,noexec,nosuid,nodev,size=64m' --cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --memory 768m --cpus 1 --user '65534:65534' --mount "type=bind,source=$stage,target=/input,readonly" $Image python -I -B /input/evaluate_rank1_humaneval.py
if ($LASTEXITCODE -ne 0) { throw "Docker evaluation failed: $LASTEXITCODE" }
$parsed=($result -join "`n") | ConvertFrom-Json
if ($null -eq $parsed.original.pass_at_1 -or $null -eq $parsed.rank1.pass_at_1) { throw 'Invalid result' }
$result | Set-Content -LiteralPath $log -Encoding utf8
$parsed.original.summary
$parsed.rank1.summary
Write-Host "Functional results: $log"
