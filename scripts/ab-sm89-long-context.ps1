[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BaselineDirectory,
    [Parameter(Mandatory)][string]$CandidateDirectory,
    [Parameter(Mandatory)][string]$CandidateRevision,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(1,5)][int]$Pairs = 2,
    [ValidateRange(2048,1048576)][int[]]$Contexts = @(32768,65536,131072),
    [ValidateRange(1,63)][int]$DraftTokens = 5,
    [ValidateSet(1024,2048,4096)][int]$PrefillChunk = 1024,
    [ValidateSet('rk4v4','rk8v4')][string]$KvDtype = 'rk4v4',
    [ValidateRange(1,2048)][int]$DecodeTokens = 128,
    [ValidateRange(2,4096)][int]$Headroom = 256,
    [switch]$ExperimentalBuckets,
    [switch]$StartupCalibration,
    [switch]$PhysicalStartupCalibration
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$BaselineDirectory = (Resolve-Path -LiteralPath $BaselineDirectory).Path
$CandidateDirectory = (Resolve-Path -LiteralPath $CandidateDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
if ($PhysicalStartupCalibration -and !$StartupCalibration) { throw 'Physical startup calibration also requires StartupCalibration' }
if ($Headroom -le $DecodeTokens -or ($Contexts | Where-Object { $_ -le $Headroom })) {
    throw 'Headroom must exceed measured decode tokens and fit every context'
}
$Contexts = @($Contexts | Sort-Object -Unique)
$corpusCount = $Contexts[-1]
if ($BaselineDirectory -eq $CandidateDirectory) { throw 'A/B requires two different builds' }
if ($CandidateRevision -notmatch '^[a-f0-9]{40}$') { throw 'CandidateRevision must be a full Git SHA' }
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Refusing to overwrite previous benchmark evidence' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

function Assert-GpuIdle {
    if (Get-Process -Name 'ninfer','ninfer-serve','ninfer_bench' -ErrorAction SilentlyContinue) {
        throw 'Another inference process is active; refusing to disrupt it'
    }
    $line = (& nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits | Select-Object -First 1)
    if ($LASTEXITCODE -ne 0 -or $line -notmatch 'RTX 4090,\s*(\d+)' -or [int]$Matches[1] -lt 19000) {
        throw 'An idle RTX 4090 with at least 19000 MiB free VRAM is required'
    }
}

$versions = @{}
$hashes = @{}
foreach ($variant in 'baseline','candidate') {
    $dir = if ($variant -eq 'baseline') { $BaselineDirectory } else { $CandidateDirectory }
    $cli = Join-Path $dir 'ninfer.exe'
    $bench = Join-Path $dir 'ninfer_bench.exe'
    $versions[$variant] = ((& $cli --version) -join ' ').Trim()
    if ($LASTEXITCODE -ne 0) { throw "Cannot start $variant CLI" }
    $hashes[$variant] = (Get-FileHash -LiteralPath $bench -Algorithm SHA256).Hash.ToLowerInvariant()
}
if ($versions.candidate -notmatch [regex]::Escape($CandidateRevision.Substring(0,7)) -or
    $versions.candidate -match 'dirty' -or $hashes.baseline -eq $hashes.candidate) {
    throw 'Candidate build identity does not match the clean requested revision'
}
$source = Join-Path (Split-Path -Parent $PSScriptRoot) 'bench/fixtures/bench_corpus.ids'
$ids = [regex]::Matches((Get-Content -LiteralPath $source -Raw), '\d+')
if ($ids.Count -eq 0) { throw 'Missing token corpus' }
$corpus = Join-Path $OutputDirectory 'long-context.ids'
$tokens = for ($i=0; $i -lt $corpusCount; $i++) { $ids[$i % $ids.Count].Value }
[IO.File]::WriteAllText($corpus, (($tokens -join ' ') + "\n").Replace('\n', [Environment]::NewLine), [Text.ASCIIEncoding]::new())
$meta = @{
    baseline_version = $versions.baseline; candidate_version = $versions.candidate
    candidate_revision = $CandidateRevision; bench_sha256 = $hashes
    model_sha256 = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
    corpus_sha256 = (Get-FileHash -LiteralPath $corpus -Algorithm SHA256).Hash.ToLowerInvariant()
    source_corpus_tokens = $ids.Count; corpus_tokens = $corpusCount; pairs = $Pairs
    contexts = $Contexts; draft_tokens = $DraftTokens; prefill_chunk = $PrefillChunk; kv_dtype = $KvDtype
    decode_tokens = $DecodeTokens; headroom = $Headroom
    experimental_buckets = [bool]$ExperimentalBuckets; startup_calibration = [bool]$StartupCalibration
    physical_startup_calibration = [bool]$PhysicalStartupCalibration
}
$meta | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'manifest.json') -Encoding utf8
$oldBuckets = $env:NINFER_SM89_ATTENTION_BUCKETS
$oldCalibration = $env:NINFER_SM89_STARTUP_CALIBRATION
$oldPhysical = $env:NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION
$results = [Collections.Generic.List[object]]::new()
try {
    foreach ($ctx in $Contexts) {
        for ($pair=1; $pair -le $Pairs; $pair++) {
            $order = if ($pair % 2 -eq 1) { 'baseline','candidate' } else { 'candidate','baseline' }
            foreach ($variant in $order) {
                Assert-GpuIdle
                $dir = if ($variant -eq 'baseline') { $BaselineDirectory } else { $CandidateDirectory }
                $env:NINFER_SM89_ATTENTION_BUCKETS = if ($variant -eq 'candidate' -and $ExperimentalBuckets) { '1' } else { '0' }
                $env:NINFER_SM89_STARTUP_CALIBRATION = if ($variant -eq 'candidate' -and $StartupCalibration) { '1' } else { '0' }
                $env:NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION = if ($variant -eq 'candidate' -and $PhysicalStartupCalibration) { '1' } else { '0' }
                $stem = "$ctx-$pair-$variant"
                $json = Join-Path $OutputDirectory "$stem.json"
                $log = Join-Path $OutputDirectory "$stem.log"
                $prompt = $ctx-$Headroom
                Write-Host "Context=$ctx Pair=$pair Variant=$variant"
                & (Join-Path $dir 'ninfer_bench.exe') --weights $Model --corpus $corpus --prompt-gen "$prompt,$DecodeTokens" --max-ctx $ctx --prefill-chunk $PrefillChunk --kv-dtype $KvDtype --gdn-state-fp16 --spec dflash2 --draft-tokens $DraftTokens --adaptive-mtp --ngram-draft-tokens 0 --warmup 0 --repetitions 1 --output json --output-file $json *> $log
                if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $json)) {
                    throw "Benchmark $stem failed; inspect $log"
                }
                $report = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
                $t = @($report.tests)
                if ($t.Count -ne 1 -or $t[0].n_prompt -ne $prompt -or $t[0].n_gen -ne $DecodeTokens -or
                    $t[0].reps[0].decode_output_tokens -ne $DecodeTokens -or $report.config.max_context -ne $ctx -or
                    $report.config.kv_cache -ne $KvDtype -or $report.config.prefill_chunk -ne $PrefillChunk -or
                    $report.config.draft_tokens -ne $DraftTokens -or $report.config.speculative_backend -ne 'dflash2') {
                    throw "Benchmark report $stem is not comparable"
                }
                $results.Add([pscustomobject]@{
                    context=$ctx; pair=$pair; variant=$variant
                    prefill_tok_s=$t[0].prefill_tok_s_mean
                    decode_tok_s=$t[0].decode_output_tok_s_mean
                    acceptance=$t[0].speculative.acceptance_rate
                    load_seconds=$report.load.load_seconds
                })
                $results | Export-Csv -LiteralPath (Join-Path $OutputDirectory 'samples.csv') -NoTypeInformation
            }
        }
    }
    $summary = foreach ($ctx in $Contexts) {
        $a=@($results | Where-Object { $_.context -eq $ctx -and $_.variant -eq 'baseline' })
        $b=@($results | Where-Object { $_.context -eq $ctx -and $_.variant -eq 'candidate' })
        $ad=($a | Measure-Object decode_tok_s -Average).Average
        $bd=($b | Measure-Object decode_tok_s -Average).Average
        $ap=($a | Measure-Object prefill_tok_s -Average).Average
        $bp=($b | Measure-Object prefill_tok_s -Average).Average
        [pscustomobject]@{ context=$ctx; pairs=$Pairs; baseline_decode=$ad; candidate_decode=$bd
            decode_change_pct=100*($bd/$ad-1); baseline_prefill=$ap; candidate_prefill=$bp
            prefill_change_pct=100*($bp/$ap-1) }
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding utf8
    $summary | Format-Table -AutoSize
} finally {
    $env:NINFER_SM89_ATTENTION_BUCKETS = $oldBuckets
    $env:NINFER_SM89_STARTUP_CALIBRATION = $oldCalibration
    $env:NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION = $oldPhysical
}
