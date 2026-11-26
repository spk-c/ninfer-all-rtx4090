# Windows PowerShell 7+ WinForms launcher. No model downloads or global process termination.
[CmdletBinding()]
param([string]$RuntimeDir,[string]$ModelPath,[switch]$ValidateOnly)
$ErrorActionPreference='Stop'
if ($PSVersionTable.PSVersion.Major -lt 7 -or !$IsWindows) {throw 'Windows PowerShell 7+ required.'}
function New-LaunchSpec([string]$Dir,[string]$Model,[int]$Context,[int]$Port,[string]$KV,[string]$Backend,[int]$Draft) {
    $exe = Join-Path ([IO.Path]::GetFullPath($Dir)) 'ninfer-serve.exe'
    if (!(Test-Path -LiteralPath $exe -PathType Leaf)) {throw 'Select the extracted runtime folder.'}
    if (!(Test-Path -LiteralPath $Model -PathType Leaf) -or [IO.Path]::GetExtension($Model) -ne '.ninfer') {throw 'Select an existing .ninfer model.'}
    if ($Context -lt 1024 -or $Context -gt 262144 -or $Port -lt 1024 -or $Port -gt 65535) {throw 'Invalid context or port.'}
    if ($KV -notin @('rk4v4','rk8v4','int8','bf16') -or $Backend -notin @('none','mtp','dflash2') -or $Draft -lt 1 -or $Draft -gt 15) {throw 'Invalid cache or draft selection.'}
    if ($Backend -eq 'dflash2' -and [IO.Path]::GetFileName($Model) -match 'Bonsai-2\.5') {throw 'Bonsai 2.5 has no bundled DFlash2 adapter.'}
    $opts=@([IO.Path]::GetFullPath($Model),'--host','127.0.0.1','--port',[string]$Port,'--model-id','local-model','--max-context',[string]$Context,'--kv-capacity',[string]$Context,'--max-concurrency','1','--kv-dtype',$KV,'--spec',$Backend)
    if($Backend -ne 'none'){$opts+=@('--draft-tokens',[string]$Draft)}
    [pscustomobject]@{Executable=$exe;WorkingDirectory=(Split-Path $exe);Arguments=$opts}
}
if($ValidateOnly){
    New-LaunchSpec $RuntimeDir $ModelPath 32768 8080 'rk4v4' 'none' 5 | ConvertTo-Json -Depth 4
    return
}
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$form=[Windows.Forms.Form]::new()
$form.Text='NInfer RTX 4090 Launcher';$form.Size=[Drawing.Size]::new(700,475);$form.StartPosition='CenterScreen';$form.AutoScaleMode='Dpi'
function Add-Label($text,$x,$y){$l=[Windows.Forms.Label]::new();$l.Text=$text;$l.Location=[Drawing.Point]::new($x,$y);$l.Size=[Drawing.Size]::new(170,25);$form.Controls.Add($l)}
function Add-Box($value,$x,$y,$w){$t=[Windows.Forms.TextBox]::new();$t.Text=$value;$t.Location=[Drawing.Point]::new($x,$y);$t.Width=$w;$form.Controls.Add($t);return $t}
function Add-Button($text,$x,$y,$w){$v=[Windows.Forms.Button]::new();$v.Text=$text;$v.Location=[Drawing.Point]::new($x,$y);$v.Size=[Drawing.Size]::new($w,30);$form.Controls.Add($v);return $v}
function Add-Combo($items,$x,$y){$v=[Windows.Forms.ComboBox]::new();$v.DropDownStyle='DropDownList';$v.Items.AddRange([object[]]$items);$v.SelectedIndex=0;$v.Location=[Drawing.Point]::new($x,$y);$v.Width=180;$form.Controls.Add($v);return $v}
Add-Label 'Runtime folder' 15 20;$dir=Add-Box $RuntimeDir 175 20 390;$pickDir=Add-Button 'Browse' 575 17 95
Add-Label 'Model file' 15 60;$model=Add-Box $ModelPath 175 60 390;$pickModel=Add-Button 'Browse' 575 57 95
Add-Label 'Context tokens' 15 100;$ctx=Add-Box '32768' 175 100 180
Add-Label 'Local port' 15 140;$port=Add-Box '8080' 175 140 180
Add-Label 'KV cache' 15 180;$kv=Add-Combo @('rk4v4','rk8v4','int8','bf16') 175 180
Add-Label 'Backend' 15 220;$spec=Add-Combo @('none','mtp','dflash2') 175 220
Add-Label 'Draft tokens' 15 260;$draft=Add-Box '5' 175 260 180
$note=[Windows.Forms.Label]::new();$note.Text='Select only a backend that your artifact supports. VRAM fit is not guaranteed. Server listens on localhost.';$note.Location=[Drawing.Point]::new(15,295);$note.Size=[Drawing.Size]::new(650,42);$form.Controls.Add($note)
$start=Add-Button 'Start server' 15 350 145;$stop=Add-Button 'Stop launched server' 170 350 165;$stop.Enabled=$false
$status=[Windows.Forms.Label]::new();$status.Location=[Drawing.Point]::new(15,390);$status.Size=[Drawing.Size]::new(650,35);$status.Text='Ready';$form.Controls.Add($status)
$script:launched=$null
$pickDir.Add_Click({$d=[Windows.Forms.FolderBrowserDialog]::new();try{if($d.ShowDialog() -eq 'OK'){$dir.Text=$d.SelectedPath}}finally{$d.Dispose()}})
$pickModel.Add_Click({$d=[Windows.Forms.OpenFileDialog]::new();$d.Filter='NInfer model (*.ninfer)|*.ninfer';try{if($d.ShowDialog() -eq 'OK'){$model.Text=$d.FileName}}finally{$d.Dispose()}})
$start.Add_Click({
    try{
        if($script:launched -and !$script:launched.HasExited){throw 'The server launched by this window is already running.'}
        $launch=New-LaunchSpec $dir.Text $model.Text ([int]$ctx.Text) ([int]$port.Text) ([string]$kv.SelectedItem) ([string]$spec.SelectedItem) ([int]$draft.Text)
        if(Get-NetTCPConnection -LocalPort ([int]$port.Text) -State Listen -ErrorAction SilentlyContinue){throw 'Port already in use.'}
        $si=[Diagnostics.ProcessStartInfo]::new();$si.FileName=$launch.Executable;$si.WorkingDirectory=$launch.WorkingDirectory;$si.UseShellExecute=$false
        foreach($arg in $launch.Arguments){$si.ArgumentList.Add($arg)}
        $script:launched=[Diagnostics.Process]::Start($si);$status.Text="Started PID $($script:launched.Id); API http://127.0.0.1:$($port.Text)/v1 (loading may take time)";$stop.Enabled=$true
    }catch{[void][Windows.Forms.MessageBox]::Show($_.Exception.Message,'Launch failed')}
})
$stop.Add_Click({if($script:launched -and !$script:launched.HasExited){$script:launched.Kill()};$status.Text='Launched server stopped.';$stop.Enabled=$false})
$form.Add_FormClosing({if($script:launched -and !$script:launched.HasExited){$answer=[Windows.Forms.MessageBox]::Show('Leave the server running? Yes keeps it running, No stops this launched process.','Close launcher','YesNoCancel');if($answer -eq 'Cancel'){$_.Cancel=$true}elseif($answer -eq 'No'){$script:launched.Kill()}}})
[void]$form.ShowDialog()
$form.Dispose()
