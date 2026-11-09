param(
    [switch]$PreviewOnly,
    [string]$Destination = (Join-Path (Split-Path $PSScriptRoot -Parent) 'results/gsle')
)
$ErrorActionPreference = 'Stop'
$serverRoot = '/mnt/sda/ubuntu/lkx/papers/paper_eli'
$sshAlias = 'ubuntuA6000'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$destinationRoot = (Resolve-Path -LiteralPath $Destination).Path
$manifestPath = Join-Path $destinationRoot 'deliverables.json'
& scp -o BatchMode=yes -o ClearAllForwardings=yes "${sshAlias}:$serverRoot/records/gsle/deliverables.json" $manifestPath
if ($LASTEXITCODE -ne 0) { throw 'Could not download the reviewed result index.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$verified = @()
foreach ($artifact in $manifest.artifacts) {
    if ($PreviewOnly -and -not $artifact.preview) { continue }
    if ([IO.Path]::GetFileName($artifact.filename) -ne $artifact.filename) { throw 'Unexpected output filename.' }
    if (-not $artifact.server_path.StartsWith("$serverRoot/outputs/gsle/")) { throw 'Unexpected source path.' }
    $targetPath = Join-Path $destinationRoot $artifact.filename
    $alreadyVerified = (Test-Path -LiteralPath $targetPath) -and ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $artifact.sha256)
    if (-not $alreadyVerified) {
        & scp -o BatchMode=yes -o ClearAllForwardings=yes "${sshAlias}:$($artifact.server_path)" $targetPath
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($artifact.filename)" }
    }
    $actual = (Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $artifact.sha256) { throw "SHA256 mismatch: $($artifact.filename)" }
    $verified += [pscustomobject]@{ file = $targetPath; sha256 = $actual; bytes = (Get-Item -LiteralPath $targetPath).Length }
    Write-Output "Verified $($artifact.filename)"
}
$verificationName = if ($PreviewOnly) { 'preview_download_verification.json' } else { 'download_verification.json' }
$verified | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destinationRoot $verificationName) -Encoding utf8
