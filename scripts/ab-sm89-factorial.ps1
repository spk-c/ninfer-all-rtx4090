[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BaselineDirectory,
    [Parameter(Mandatory)][string]$CandidateDirectory,
    [Parameter(Mandatory)][string]$CandidateRevision,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Corpus,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(2,5)][int]$Repetitions = 2
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$BaselineDirectory = (Resolve-Path -LiteralPath $BaselineDirectory).Path
$CandidateDirectory = (Resolve-Path -LiteralPath $CandidateDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$Corpus = (Resolve-Path -LiteralPath $Corpus).Path
if ($BaselineDirectory -eq $CandidateDirectory -or $CandidateRevision -notmatch '^[a-f0-9]{40}$') {
    throw 'Separate binaries and a full candidate SHA are required'
}
$versions=@{}
$hashes=@{}
foreach ($name in 'baseline','candidate') {
    $dir=if($name -eq 'baseline'){$BaselineDirectory}else{$CandidateDirectory}
    $versions[$name]=((& (Join-Path $dir 'ninfer.exe') --version) -join ' ').Trim()
    if($LASTEXITCODE -ne 0){throw "CLI identity check failed: $name"}
    $hashes[$name]=(Get-FileHash -LiteralPath (Join-Path $dir 'ninfer_bench.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
}
if($versions.candidate -notmatch [regex]::Escape($CandidateRevision.Substring(0,7)) -or
   $versions.candidate -match 'dirty' -or $hashes.baseline -eq $hashes.candidate){throw 'Candidate identity mismatch'}
if(Test-Path -LiteralPath $OutputDirectory){throw 'Refusing to overwrite previous measurements'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path -LiteralPath $OutputDirectory).Path
$arms=@(
    @{name='baseline';candidate=$false;buckets='0';calibration='0'},
    @{name='control';candidate=$true;buckets='0';calibration='0'},
    @{name='buckets';candidate=$true;buckets='1';calibration='0'},
    @{name='calibration';candidate=$true;buckets='0';calibration='1'},
    @{name='both';candidate=$true;buckets='1';calibration='1'}
)
$manifest=@{
    versions=$versions; bench_sha256=$hashes; candidate_revision=$CandidateRevision
    model_sha256=(Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
    corpus_sha256=(Get-FileHash -LiteralPath $Corpus -Algorithm SHA256).Hash.ToLowerInvariant()
    repetitions=$Repetitions; arms=$arms; decode_tokens=512; prompt_headroom=1024
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$OutputDirectory\manifest.json" -Encoding utf8
$oldBuckets=$env:NINFER_SM89_ATTENTION_BUCKETS
$oldCalibration=$env:NINFER_SM89_STARTUP_CALIBRATION
$results=[Collections.Generic.List[object]]::new()
try {
    foreach($ctx in 32768,65536,131072) {
        for($rep=1;$rep -le $Repetitions;$rep++){
            $order=@($arms)
            if($rep % 2 -eq 0){[array]::Reverse($order)}
            foreach($arm in $order){
                if(Get-Process -Name 'ninfer','ninfer-serve','ninfer_bench' -ErrorAction SilentlyContinue){throw 'Other inference process active'}
                $gpu=(& nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits | Select-Object -First 1)
                if($LASTEXITCODE -ne 0 -or $gpu -notmatch 'RTX 4090,\s*(\d+)' -or [int]$Matches[1] -lt 19000){throw 'RTX 4090 requires 19000 MiB free'}
                $dir=if($arm.candidate){$CandidateDirectory}else{$BaselineDirectory}
                $env:NINFER_SM89_ATTENTION_BUCKETS=$arm.buckets
                $env:NINFER_SM89_STARTUP_CALIBRATION=$arm.calibration
                $name=$arm.name
                $stem="$ctx-$rep-$name"
                $json=Join-Path $OutputDirectory "$stem.json"
                $prompt=$ctx-1024
                Write-Host "Context=$ctx Rep=$rep Arm=$name"
                & (Join-Path $dir 'ninfer_bench.exe') --weights $Model --corpus $Corpus --prompt-gen "$prompt,512" --max-ctx $ctx --prefill-chunk 1024 --kv-dtype rk4v4 --gdn-state-fp16 --spec dflash2 --draft-tokens 5 --adaptive-mtp --ngram-draft-tokens 0 --warmup 0 --repetitions 1 --output json --output-file $json *> (Join-Path $OutputDirectory "$stem.log")
                if($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $json)){throw "Failed: $stem"}
                $report=Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
                $t=@($report.tests)
                if($t.Count -ne 1 -or $t[0].n_prompt -ne $prompt -or $t[0].n_gen -ne 512 -or
                   $t[0].reps[0].decode_output_tokens -ne 512 -or $report.config.max_context -ne $ctx -or
                   $report.config.kv_cache -ne 'rk4v4' -or $report.config.speculative_backend -ne 'dflash2'){
                    throw "Incomparable report: $stem"
                }
                $results.Add([pscustomobject]@{
                    context=$ctx; repetition=$rep; arm=$name
                    prefill_tok_s=$t[0].prefill_tok_s_mean; decode_tok_s=$t[0].decode_output_tok_s_mean
                    acceptance=$t[0].speculative.acceptance_rate; drafted=$t[0].speculative.drafted_tokens
                    load_seconds=$report.load.load_seconds
                })
                $results | Export-Csv -LiteralPath "$OutputDirectory\samples.csv" -NoTypeInformation
                $results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$OutputDirectory\samples.json" -Encoding utf8
            }
        }
    }
    $summary=foreach($ctx in 32768,65536,131072){
        $group=@($results | Where-Object context -eq $ctx)
        $base=($group | Where-Object arm -eq 'baseline' | Measure-Object decode_tok_s -Average).Average
        $control=($group | Where-Object arm -eq 'control' | Measure-Object decode_tok_s -Average).Average
        foreach($arm in $arms){
            $name=$arm.name
            $samples=@($group | Where-Object arm -eq $name)
            if($samples.Count -ne $Repetitions){throw 'Incomplete factorial matrix'}
            $mean=($samples | Measure-Object decode_tok_s -Average).Average
            [pscustomobject]@{
                context=$ctx; arm=$name; samples=$samples.Count
                decode_tok_s=$mean
                vs_baseline_pct=100*($mean/$base-1)
                vs_control_pct=100*($mean/$control-1)
                prefill_tok_s=($samples | Measure-Object prefill_tok_s -Average).Average
                acceptance=($samples | Measure-Object acceptance -Average).Average
            }
        }
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$OutputDirectory\summary.json" -Encoding utf8
    $summary | Format-Table -AutoSize
} finally {
    $env:NINFER_SM89_ATTENTION_BUCKETS=$oldBuckets
    $env:NINFER_SM89_STARTUP_CALIBRATION=$oldCalibration
}
