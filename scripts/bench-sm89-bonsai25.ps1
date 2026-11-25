[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$CandidateDirectory,
    [Parameter(Mandatory)][string]$CandidateRevision,
    [Parameter(Mandatory)][string]$ModelDirectory,
    [Parameter(Mandatory)][string]$Corpus,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateRange(2,5)][int]$Repetitions=2
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$CandidateDirectory=(Resolve-Path $CandidateDirectory).Path
$ModelDirectory=(Resolve-Path $ModelDirectory).Path
$Corpus=(Resolve-Path $Corpus).Path
$version=((& (Join-Path $CandidateDirectory 'ninfer.exe') --version) -join ' ').Trim()
if($LASTEXITCODE -ne 0 -or $CandidateRevision -notmatch '^[a-f0-9]{40}$' -or $version -notmatch [regex]::Escape($CandidateRevision.Substring(0,9)) -or $version -match 'dirty'){throw 'Candidate identity mismatch'}
if(Test-Path $OutputDirectory){throw 'Output directory already exists'}
New-Item -ItemType Directory $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path $OutputDirectory).Path
$models=@{
    ptq=@{name='Ternary-Bonsai-2.5-27B-PTQ1_0-ninfer-v3.ninfer';sha='76cc0056489a279d046b0395deaa6840ad4ca1757dfd509c915e84c2d64237ed'}
    pq=@{name='Ternary-Bonsai-2.5-27B-PQ2_0-ninfer-v3.ninfer';sha='476ed21eb4c0a99fee491eff111257d00514188c3e304a7b1fef7fd5941dc6b7'}
}
foreach($model in $models.Values){
    $model.path=Join-Path $ModelDirectory $model.name
    if((Get-FileHash $model.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $model.sha){throw 'Model hash mismatch'}
}
$prompts=@(1024,31744,64512,130048,175104,195584)
$window=262144
$arms=@(
    @{name='ptq-plain';model='ptq';mtp=$false},
    @{name='pq-plain';model='pq';mtp=$false},
    @{name='ptq-mtp3';model='ptq';mtp=$true},
    @{name='pq-mtp3';model='pq';mtp=$true}
)
$manifest=@{version=$version;candidate_revision=$CandidateRevision;bench_sha256=(Get-FileHash (Join-Path $CandidateDirectory 'ninfer_bench.exe')).Hash.ToLowerInvariant();models=$models;corpus_sha256=(Get-FileHash $Corpus).Hash.ToLowerInvariant();prompts=$prompts;max_context=$window;decode_tokens=512;kv_dtype='rk8v4';prefill_chunk=1024;repetitions=$Repetitions;arms=$arms;telemetry='nvidia-smi 500ms; load, prefill and decode combined'}
$manifest | ConvertTo-Json -Depth 8 | Set-Content "$OutputDirectory\manifest.json" -Encoding utf8
$flags=@('NINFER_SM89_ATTENTION_BUCKETS','NINFER_SM89_FLAT_DRAFT_GRAPHS','NINFER_SM89_FLAT_TARGET_GRAPHS','NINFER_SM89_STARTUP_CALIBRATION','NINFER_SM89_PHYSICAL_STARTUP_CALIBRATION','NINFER_SM89_LIVE_CONTEXT_REPLAY')
$old=@{}
$results=[Collections.Generic.List[object]]::new()
try{
    foreach($flag in $flags){$old[$flag]=[Environment]::GetEnvironmentVariable($flag,'Process');[Environment]::SetEnvironmentVariable($flag,'0','Process')}
    for($rep=1;$rep -le $Repetitions;$rep++){
        $order=@($arms);if($rep % 2 -eq 0){[array]::Reverse($order)}
        foreach($arm in $order){
            if(Get-Process -Name 'ninfer','ninfer-serve','ninfer_bench','ninfer_qwen3_5_pressure_replay_real_test','ninfer_qwen3_5_flat_model_real_test' -ErrorAction SilentlyContinue){throw 'Other inference process active'}
            $gpu=(& nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits | Select-Object -First 1)
            if($LASTEXITCODE -ne 0 -or $gpu -notmatch 'RTX 4090,\s*(\d+)' -or [int]$Matches[1] -lt 19000){throw 'GPU not ready'}
            $stem="$rep-$($arm.name)"
            $json="$OutputDirectory\$stem.json"
            $pg=($prompts | ForEach-Object {"$_,512"}) -join ';'
            $argsList=@('--weights',$models[$arm.model].path,'--corpus',$Corpus,'--prompt-gen',$pg,'--max-ctx',"$window",'--prefill-chunk','1024','--kv-dtype','rk8v4','--gdn-state-fp16','--ngram-draft-tokens','0','--warmup','0','--repetitions','1','--output','json','--output-file',$json)
            if($arm.mtp){$argsList+=@('--spec','mtp','--draft-tokens','3')}
            $telemetry=Start-Process -FilePath (Get-Command nvidia-smi.exe).Source -ArgumentList @('--query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used','--format=csv,nounits','-lms','500') -RedirectStandardOutput "$OutputDirectory\$stem.gpu.csv" -RedirectStandardError "$OutputDirectory\$stem.gpu.err" -PassThru -WindowStyle Hidden
            try {
                Write-Output "Starting $stem"
                & (Join-Path $CandidateDirectory 'ninfer_bench.exe') @argsList *> "$OutputDirectory\$stem.log"
                $rc=$LASTEXITCODE
            }finally{
                if(!$telemetry.HasExited){Stop-Process -Id $telemetry.Id -ErrorAction SilentlyContinue}
                $telemetry.Dispose()
            }
            if($rc -ne 0 -or !(Test-Path $json)){throw "Benchmark failed: $stem"}
            $report=Get-Content $json -Raw | ConvertFrom-Json
            $backend=if($arm.mtp){'mtp'}else{'none'}
            if($report.config.speculative_backend -ne $backend -or $report.config.max_context -ne $window -or $report.config.kv_cache -ne 'rk8v4' -or @($report.tests).Count -ne $prompts.Count){throw "Report mismatch: $stem"}
            if($arm.mtp -and $report.config.draft_tokens -ne 3){throw 'Wrong draft window'}
            foreach($prompt in $prompts){
                $t=@($report.tests | Where-Object n_prompt -eq $prompt)
                if($t.Count -ne 1 -or $t[0].n_gen -ne 512 -or $t[0].reps[0].decode_output_tokens -ne 512){throw "Incomplete test: $stem $prompt"}
                $results.Add([pscustomobject]@{repetition=$rep;arm=$arm.name;prompt=$prompt;decode_tok_s=$t[0].decode_output_tok_s_mean;prefill_tok_s=$t[0].prefill_tok_s_mean;acceptance=$t[0].speculative.acceptance_rate;route=$report.config.decode_path;first_token_seconds=$t[0].reps[0].timings.first_token_seconds;load_seconds=$report.load.load_seconds})
            }
            $results | Export-Csv "$OutputDirectory\samples.csv" -NoTypeInformation
            ConvertTo-Json -InputObject @($results.ToArray()) -Depth 4 | Set-Content "$OutputDirectory\samples.json" -Encoding utf8
        }
    }
    $summary=foreach($prompt in $prompts){foreach($arm in $arms){
        $rows=@($results | Where-Object {$_.prompt -eq $prompt -and $_.arm -eq $arm.name})
        if($rows.Count -ne $Repetitions){throw 'Incomplete matrix'}
        [pscustomobject]@{prompt=$prompt;arm=$arm.name;repetitions=$rows.Count;decode_tok_s=($rows | Measure-Object decode_tok_s -Average).Average;min_decode=($rows | Measure-Object decode_tok_s -Minimum).Minimum;max_decode=($rows | Measure-Object decode_tok_s -Maximum).Maximum;prefill_tok_s=($rows | Measure-Object prefill_tok_s -Average).Average;acceptance=($rows | Measure-Object acceptance -Average).Average}
    }}
    ConvertTo-Json -InputObject @($summary) -Depth 5 | Set-Content "$OutputDirectory\summary.json" -Encoding utf8
    $summary | Format-Table -AutoSize
}finally{
    foreach($flag in $flags){[Environment]::SetEnvironmentVariable($flag,$old[$flag],'Process')}
}
