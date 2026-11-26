# Run interactively as the repository owner using the normal GitHub CLI login.
[CmdletBinding(SupportsShouldProcess)]
param()
$ErrorActionPreference='Stop'
$owner=(& gh api user --jq '.login' | Out-String).Trim()
if($LASTEXITCODE -ne 0 -or $owner -ne 'spk-c'){throw 'Run gh auth login as spk-c, then retry.'}
$r='repos/spk-c/ninfer-all-rtx4090'
$homepage='https://spk-c.github.io/ninfer-all-rtx4090/'
$desc='Native RTX 4090 (SM89) CUDA 13.4 local LLM inference engine. Qwen, Bonsai, GGUF-derived artifacts, DFlash2, MTP, OpenAI-compatible API and portable Windows binaries.'
$topics=@('rtx4090','llm','cuda','local-llm','local-ai','llm-inference','llm-serving','nvidia','gguf','qwen','dflash2','speculative-decoding','cuda-graphs','gpu-optimization','ada-lovelace','sm89','quantization','openai-compatible','windows','benchmarking')
if($PSCmdlet.ShouldProcess($r,'Set homepage, description, enable Discussions and replace topics')){
    @{homepage=$homepage;description=$desc;has_discussions=$true}|ConvertTo-Json -Compress|gh api --method PATCH $r --input - | Out-Null
    if($LASTEXITCODE -ne 0){throw 'Repository metadata update failed; check administrative permissions.'}
    @{names=$topics}|ConvertTo-Json -Compress|gh api --method PUT ($r+'/topics') --input -|Out-Null
    if($LASTEXITCODE -ne 0){throw 'Topic update failed; homepage/description may already be changed.'}
    $result=gh api $r|ConvertFrom-Json
    if($LASTEXITCODE -ne 0 -or $result.homepage -ne $homepage -or $result.description -ne $desc -or !$result.has_discussions -or @($topics|Where-Object {$_ -notin $result.topics}).Count -ne 0 -or $result.topics.Count -ne $topics.Count){throw 'Read-back verification failed.'}
    Write-Output 'Repository homepage, description, Discussions and 20 topics verified.'
    Write-Output 'Social preview still requires the Settings upload UI.'
}
