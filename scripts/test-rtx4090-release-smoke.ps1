# Test the actual portable package on a free RTX 4090 before publishing it.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PackageDirectory,
    [Parameter(Mandatory)][string]$Model,
    [ValidateRange(1024,65535)][int]$Port = 18091
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$server = $null
$exe = Join-Path $PackageDirectory 'ninfer-serve.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing portable server: $exe" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Missing test model: $Model" }
if (@(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue).Count -gt 0) {
    throw 'Another ninfer-serve is active; do not interrupt user workloads. Retry when idle.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Smoke test port $Port is already occupied"
}
$gpuNames = @(& nvidia-smi --query-gpu=name --format=csv,noheader)
if ($LASTEXITCODE -ne 0 -or -not ($gpuNames | Where-Object { $_ -match 'RTX 4090' })) {
    throw 'A local RTX 4090 is required to qualify the release'
}
$freeMiB = [int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1).Trim())
if ($freeMiB -lt 19000) {
    throw "Insufficient VRAM ($freeMiB MiB). Retry when the RTX 4090 is idle."
}
try {
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $exe
    $psi.WorkingDirectory = $PackageDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $args = @($Model, '--model-id','release-smoke', '--host','127.0.0.1',
        '--port', [string]$Port, '--max-context','4096', '--kv-capacity','auto',
        '--kv-dtype','rk4v4', '--gdn-state-fp16', '--max-concurrency','1',
        '--spec','dflash2', '--draft-tokens','5')
    foreach ($arg in $args) { [void]$psi.ArgumentList.Add($arg) }
    $server = [Diagnostics.Process]::Start($psi)
    if (-not $server) { throw 'Portable server could not start' }
    $ready = $false
    for ($i=0; $i -lt 90; $i++) {
        if ($server.HasExited) { throw "Server exited during load: $($server.ExitCode)" }
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3
            $ready = $true
            break
        } catch { Start-Sleep -Seconds 2 }
    }
    if (-not $ready) { throw 'Portable server never became ready' }
    # Small budgets may consume all tokens in Qwen reasoning_content, yielding empty content.
    $body = @{ model='release-smoke'; messages=@(@{role='user'; content='Say hello.'});
               max_tokens=256 } | ConvertTo-Json -Depth 6
    $reply = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 180
    if (-not $reply.choices -or $reply.choices[0].finish_reason -ne 'stop' -or
        [string]::IsNullOrWhiteSpace($reply.choices[0].message.content)) {
        throw "GPU smoke incomplete (finish_reason=$($reply.choices[0].finish_reason))"
    }
    Write-Host "RTX4090 DFlash2 real-model smoke passed: $($reply.choices[0].message.content)"
} finally {
    if ($server) {
        if (-not $server.HasExited) {
            $server.Kill()
            [void]$server.WaitForExit(10000)
        }
        $server.Dispose()
    }
}
