# Requires PowerShell 7. No administrator access, PATH changes or model downloads.
# Installs the current stable native SM89 archive into a new versioned directory.
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'NInfer\RTX4090'),
    [string]$ArchiveCache,
    [string]$SevenZip
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7 -or !$IsWindows) { throw 'Windows PowerShell 7+ is required.' }
if (!$SevenZip) {
    $z = Get-Command 7z.exe -ErrorAction SilentlyContinue
    $SevenZip = if ($z) { $z.Source } else { Join-Path $env:ProgramFiles '7-Zip\7z.exe' }
}
if (!(Test-Path -LiteralPath $SevenZip -PathType Leaf)) { throw 'Install 7-Zip or pass -SevenZip.' }
$headers = @{ Accept='application/vnd.github+json'; 'User-Agent'='NInfer-RTX4090-Installer' }
$api = 'https://api.github.com/repos/spk-c/ninfer-all-rtx4090/releases/latest'
$release = Invoke-RestMethod $api -Headers $headers -TimeoutSec 30
$tag = [string]$release.tag_name
if ($release.draft -or $release.prerelease -or $tag -notmatch '^rtx4090-v\d+\.\d+\.\d+$') { throw 'Latest release is not a stable RTX4090 package.' }
$volumes = @($release.assets | Where-Object { $_.name -match '^ninfer-windows-x64-cuda13\.4-sm89-[0-9a-f]{12}\.7z\.\d{3}$' } | Sort-Object name)
$manifests = @($release.assets | Where-Object { $_.name -match '^SHA256SUMS-ninfer-windows-x64-cuda13\.4-sm89-[0-9a-f]{12}\.txt$' })
if ($volumes.Count -lt 1 -or $manifests.Count -ne 1 -or $volumes[0].name -notmatch '\.001$') { throw 'Unexpected release asset layout.' }
$prefix = $volumes[0].name -replace '\.7z\.001$',''
if ($manifests[0].name -ne ('SHA256SUMS-'+$prefix+'.txt')) { throw 'Manifest belongs to another package.' }
for ($i=0;$i -lt $volumes.Count;$i++) {
    if ($volumes[$i].name -ne ($prefix+'.7z.'+($i+1).ToString('000'))) { throw 'Unexpected or missing archive volume.' }
}
$root = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $root | Out-Null
$final = Join-Path $root $tag
if (Test-Path -LiteralPath $final) { throw "Version already exists: $final. No overwrite." }
$stage = Join-Path $root ('.install-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stage | Out-Null
try {
    foreach ($asset in @($manifests[0]) + $volumes) {
        if ($asset.digest -notmatch '^sha256:([a-fA-F0-9]{64})$') { throw "Missing digest: $($asset.name)" }
        $digest = $Matches[1].ToLowerInvariant()
        $path = Join-Path $stage $asset.name
        $cache = if ($ArchiveCache) { Join-Path $ArchiveCache $asset.name } else { $null }
        if ($cache -and (Test-Path -LiteralPath $cache -PathType Leaf)) { Copy-Item -LiteralPath $cache -Destination $path }
        else {
            if (!$asset.browser_download_url.StartsWith('https://github.com/spk-c/ninfer-all-rtx4090/releases/download/')) { throw 'Unexpected asset URL.' }
            Invoke-WebRequest $asset.browser_download_url -OutFile $path -TimeoutSec 1800
        }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) { throw "GitHub digest mismatch: $($asset.name)" }
        if ((Get-Item -LiteralPath $path).Length -ne $asset.size) { throw "Size mismatch: $($asset.name)" }
    }
    $checks = @{}
    foreach ($line in Get-Content -LiteralPath (Join-Path $stage $manifests[0].name)) {
        if ($line -match '^([a-fA-F0-9]{64})\s+\*?(.+)$') { $checks[$Matches[2].Trim()] = $Matches[1].ToLowerInvariant() }
    }
    foreach ($asset in $volumes) {
        if (!$checks.ContainsKey($asset.name) -or $checks[$asset.name] -ne (Get-FileHash -LiteralPath (Join-Path $stage $asset.name) -Algorithm SHA256).Hash.ToLowerInvariant()) { throw "Manifest mismatch: $($asset.name)" }
    }
    $archive = Join-Path $stage $volumes[0].name
    & $SevenZip t $archive | Out-Host
    if ($LASTEXITCODE -ne 0) { throw '7-Zip integrity check failed.' }
    $extract = Join-Path $stage 'extract'
    & $SevenZip x $archive ('-o'+$extract) -y | Out-Host
    if ($LASTEXITCODE -ne 0) { throw '7-Zip extraction failed.' }
    $servers = @(Get-ChildItem -LiteralPath $extract -Filter ninfer-serve.exe -File -Recurse)
    if ($servers.Count -ne 1) { throw 'Unexpected executable layout.' }
    $runtime = $servers[0].Directory.FullName
    $info = Get-Content -LiteralPath (Join-Path $runtime 'BUILD-INFO.json') -Raw | ConvertFrom-Json
    if ($info.version -ne $tag -or $info.modified_sources -ne $false -or $info.cuda -ne '13.4' -or '89' -notin @($info.architectures | ForEach-Object {[string]$_})) { throw 'Build metadata mismatch.' }
    $version = (& $servers[0].FullName --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $version -notmatch [regex]::Escape($tag)) { throw 'Binary version mismatch.' }
    Move-Item -LiteralPath $runtime -Destination $final
    Write-Output "Installed verified stable $tag at $final"
    Write-Output 'Model weights are separate. See https://spk-c.github.io/ninfer-all-rtx4090/configurator.html'
} finally {
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
}
