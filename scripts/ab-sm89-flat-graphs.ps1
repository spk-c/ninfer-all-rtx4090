[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$CandidateDirectory,
    [Parameter(Mandatory)][string]$CandidateRevision,
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Corpus,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(2,5)][int]$Repetitions = 2,
    [ValidateRange(2048,1048576)][int[]]$Contexts = @(32768,65536,131072,176128,196608),
    [ValidateRange(1,15)][int]$DraftTokens = 5
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$CandidateDirectory = (Resolve-Path -LiteralPath $CandidateDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$Corpus = (Resolve-Path -LiteralPath $Corpus).Path
if ($CandidateRevision -notmatch '^[a-f0-9]{40}$') {throw 'Full candidate SHA required'}
$Contexts=@($Contexts | Sort-Object -Unique)
$version=((& (Join-Path $CandidateDirectory 'ninfer.exe') --version) -join ' ').Trim()
if($LASTEXITCODE -ne 0 -or $version -notmatch [regex]::Escape($CandidateRevision.Substring(0,7)) -or $version -match 'dirty'){throw 'Candidate identity mismatch'}
$hash=(Get-FileHash -LiteralPath (Join-Path $CandidateDirectory 'ninfer_bench.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
if(Test-Path -LiteralPath $OutputDirectory){throw 'Refusing to overwrite previous measurements'}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path -LiteralPath $OutputDirectory).Path
$arms=@(
    @{name='control';buckets='0';flat='0';target='0'},
    @{name='buckets';buckets='1';flat='0';target='0'},
    @{name='flat-draft';buckets='1';flat='1';target='0'},
    @{name='flat-target';buckets='1';flat='1';target='1'}
)
$manifest=@{
    version=$version; bench_sha256=$hash; candidate_revision=$CandidateRevision
    contexts=$Contexts; draft_tokens=$DraftTokens
    model_sha256=(Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
    corpus_sha256=(Get-FileHash -LiteralPath $Corpus -Algorithm SHA256).Hash.ToLowerInvariant()
    repetitions=$Repetitions; arms=$arms; decode_tokens=512; prompt_headroom=1024
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$OutputDirectory\manifest.json" -Encoding utf8
$oldBuckets=$env:NINFER_SM89_ATTENTION_BUCKETS
$oldCalibration=$env:NINFER_SM89_STARTUP_CALIBRATION
$oldPhysical=$env:NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION
$oldFlat=$env:NINFER_SM89_FLAT_DRAFT_GRAPHS
$oldTarget=$env:NINFER_SM89_FLAT_TARGET_GRAPHS
$oldRecovery=$env:NINFER_SM89_LIVE_CONTEXT_REPLAY
$results=[Collections.Generic.List[object]]::new()
try {
    foreach($ctx in $Contexts) {
        for($rep=1;$rep -le $Repetitions;$rep++){
            $order=@($arms)
            if($rep % 2 -eq 0){[array]::Reverse($order)}
            foreach($arm in $order){
                if(Get-Process -Name 'ninfer','ninfer-serve','ninfer_bench' -ErrorAction SilentlyContinue){throw 'Other inference process active'}
                $gpu=(& nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits | Select-Object -First 1)
                if($LASTEXITCODE -ne 0 -or $gpu -notmatch 'RTX 4090,\s*(\d+)' -or [int]$Matches[1] -lt 19000){throw 'RTX 4090 requires 19000 MiB free'}
                $dir=$CandidateDirectory
                $env:NINFER_SM89_ATTENTION_BUCKETS=$arm.buckets
                $env:NINFER_SM89_STARTUP_CALIBRATION='0'
                $env:NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION='0'
                $env:NINFER_SM89_FLAT_DRAFT_GRAPHS=$arm.flat
                $env:NINFER_SM89_FLAT_TARGET_GRAPHS=$arm.target
                $env:NINFER_SM89_LIVE_CONTEXT_REPLAY='0'
                $name=$arm.name
                $stem="$ctx-$rep-$name"
                $json=Join-Path $OutputDirectory "$stem.json"
                $prompt=$ctx-1024
                Write-Host "Context=$ctx Rep=$rep Arm=$name"
                & (Join-Path $dir 'ninfer_bench.exe') --weights $Model --corpus $Corpus --prompt-gen "$prompt,512" --max-ctx $ctx --prefill-chunk 1024 --kv-dtype rk4v4 --gdn-state-fp16 --spec dflash2 --draft-tokens $DraftTokens --adaptive-mtp --ngram-draft-tokens 0 --warmup 0 --repetitions 1 --output json --output-file $json *> (Join-Path $OutputDirectory "$stem.log")
                if($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $json)){throw "Failed: $stem"}
                $report=Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
                $t=@($report.tests)
                if($t.Count -ne 1 -or $t[0].n_prompt -ne $prompt -or $t[0].n_gen -ne 512 -or
                   $t[0].reps[0].decode_output_tokens -ne 512 -or $report.config.max_context -ne $ctx -or
                   $report.config.kv_cache -ne 'rk4v4' -or $report.config.draft_tokens -ne $DraftTokens -or $report.config.speculative_backend -ne 'dflash2'){
                    throw "Incomparable report: $stem"
                }
                $sites=0; $profiles=0; $replays=0; $fallbacks=0
                $logtext=Get-Content -LiteralPath (Join-Path $OutputDirectory "$stem.log") -Raw
                if($arm.flat -eq '1'){
                    foreach($m in [regex]::Matches($logtext,'SM89 flat draft graph .*?: (\d+) profiles, (\d+) sites, (\d+) startup replays, (\d+) fallbacks')){
                        $profiles += [int]$m.Groups[1].Value
                        $sites += [int]$m.Groups[2].Value
                        $replays += [int]$m.Groups[3].Value
                        $fallbacks += [int]$m.Groups[4].Value
                    }
                    if($sites -eq 0 -or $replays -eq 0 -or $fallbacks -ne 0 -or $logtext -match 'flat draft graph (fallback|capture rejected)'){
                        throw "Flat graph path not qualified: $stem"
                    }
                }
                $results.Add([pscustomobject]@{
                    flat_profiles=$profiles; flat_sites=$sites; flat_replays=$replays; flat_fallbacks=$fallbacks
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
    $summary=foreach($ctx in $Contexts){
        $group=@($results | Where-Object context -eq $ctx)
        $control=($group | Where-Object arm -eq 'control' | Measure-Object decode_tok_s -Average).Average
        $bucket=($group | Where-Object arm -eq 'buckets' | Measure-Object decode_tok_s -Average).Average
        foreach($arm in $arms){
            $name=$arm.name
            $samples=@($group | Where-Object arm -eq $name)
            if($samples.Count -ne $Repetitions){throw 'Incomplete factorial matrix'}
            $mean=($samples | Measure-Object decode_tok_s -Average).Average
            [pscustomobject]@{
                context=$ctx; arm=$name; samples=$samples.Count
                decode_tok_s=$mean
                vs_buckets_pct=100*($mean/$bucket-1)
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
    $env:NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION=$oldPhysical
    $env:NINFER_SM89_FLAT_DRAFT_GRAPHS=$oldFlat
    $env:NINFER_SM89_FLAT_TARGET_GRAPHS=$oldTarget
    $env:NINFER_SM89_LIVE_CONTEXT_REPLAY=$oldRecovery
}
