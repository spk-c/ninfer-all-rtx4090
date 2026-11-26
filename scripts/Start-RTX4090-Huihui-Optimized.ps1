[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BinDir,
    [Parameter(Mandatory=$true)][string]$Model,
    [ValidateRange(1,65535)][int]$Port=8080,
    [string]$LogDir=(Join-Path $PWD 'logs')
)
$ErrorActionPreference='Stop'
$exe=Join-Path $BinDir 'ninfer-serve.exe'
$config=Join-Path $PSScriptRoot '..\configs\rtx4090-huihui-dflash2-k7.ini'
foreach($file in @($exe,$Model,$config)){
    if(!(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing file: $file"}
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
& $exe $Model --config $config --port $Port --request-log-jsonl (Join-Path $LogDir 'req-huihui-k7-p4096.jsonl') --request-log-max-mib 200 --request-log-keep 3 --log-stats-interval-ms 15000
if($LASTEXITCODE -ne 0){throw "ninfer-serve exited $LASTEXITCODE"}
