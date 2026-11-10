# RTX 4090 / Windows: run the isolated Expert Delivery cases under Compute Sanitizer.
# This intentionally instruments the eager host-handshake path. The default
# Expert Delivery test still exercises CUDA Graph replay separately, without
# instrumentation. These two paths are NOT equivalent sanitizer coverage.
[CmdletBinding()]
param(
    [ValidateSet('memcheck', 'racecheck', 'synccheck', 'all')]
    [string]$Tool = 'all',
    [string]$BuildDir = 'build-validation',
    [string]$Sanitizer = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\compute-sanitizer\compute-sanitizer.exe',
    [int]$TimeoutSeconds = 180,
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
if ($TimeoutSeconds -lt 1) { throw 'TimeoutSeconds must be positive' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (![IO.Path]::IsPathRooted($BuildDir)) { $BuildDir = Join-Path $repo $BuildDir }
$exe = Join-Path $BuildDir 'tests\ninfer_expert_delivery_test.exe'
if (!(Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing test executable: $exe" }
if (!(Test-Path -LiteralPath $Sanitizer -PathType Leaf)) { throw "Missing sanitizer executable: $Sanitizer" }
if (!$OutputDir) { $OutputDir = Join-Path $repo '.local\expert-sanitizer' }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$cases = @('admission', 'tier-policy', 'slot-pools', 'alignment')
foreach ($mode in @('ram', 'disk')) {
    foreach ($name in @('cache-0m','cache-6m','cache-64m','alternate','cpu-half',
                       'cpu-full','elastic','lent-banks','span-prefetch','lent-landing')) {
        $cases += "$mode-$name"
    }
}
$cases += 'failure-control'
$tools = if ($Tool -eq 'all') { @('memcheck','racecheck','synccheck') } else { @($Tool) }
$results = [System.Collections.Generic.List[object]]::new()

foreach ($check in $tools) {
    foreach ($case in $cases) {
        $stem = "$check-$case"
        $log = Join-Path $OutputDir "$stem.log"
        $stdout = Join-Path $OutputDir "$stem.stdout"
        $stderr = Join-Path $OutputDir "$stem.stderr"
        $psi = [Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $Sanitizer
        $psi.Arguments = "--tool $check --error-exitcode 99 --log-file `"$log`" `"$exe`" --case $case --sanitizer"
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        # Keep NInfer's default device watchdog ENABLED; the process timeout
        # below is a separate outer limit and a timeout is always a failure.
        $psi.EnvironmentVariables.Remove('NINFER_DEVICE_STALL_SECONDS')
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $proc = [Diagnostics.Process]::Start($psi)
        try {
            $outTask = $proc.StandardOutput.ReadToEndAsync()
            $errTask = $proc.StandardError.ReadToEndAsync()
            $finished = $proc.WaitForExit($TimeoutSeconds * 1000)
            if (!$finished) {
                try { $proc.Kill() } catch {}
                $proc.WaitForExit()
                $code = 124
            } else {
                $proc.WaitForExit()
                $code = $proc.ExitCode
            }
            [IO.File]::WriteAllText($stdout, $outTask.Result)
            [IO.File]::WriteAllText($stderr, $errTask.Result)
        } finally {
            $watch.Stop()
            $proc.Dispose()
        }
        $output = [IO.File]::ReadAllText($stdout)
        $sanlog = if (Test-Path $log) { [IO.File]::ReadAllText($log) } else { '' }
        $assertions = $output.Contains("PASS expert-delivery case $case (exit 0)")
        $clean = if ($check -eq 'racecheck') {
            $sanlog.Contains('0 hazards displayed (0 errors, 0 warnings)')
        } else {
            $sanlog.Contains('ERROR SUMMARY: 0 errors')
        }
        $success = $code -eq 0 -and $assertions -and $clean
        $entry = [pscustomobject]@{
            tool = $check; case = $case; result = if ($success) { 'PASS' } elseif ($code -eq 124) { 'TIMEOUT' } else { 'FAIL' }
            exit = $code; assertions = $assertions; sanitizer_clean = $clean
            seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 2)
        }
        $results.Add($entry)
        $results | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $OutputDir 'results.json')
        Write-Host "$($entry.result) $stem exit=$code seconds=$($entry.seconds) assertions=$assertions clean=$clean"
        if (!$success) { throw "Expert sanitizer failed: $stem; inspect $log, $stdout and $stderr" }
    }
}
Write-Host "EXPERT_SANITIZER_PASS tools=$($tools.Count) cases_per_tool=$($cases.Count) total=$($results.Count)"
