param([int]$Port = 18086)
$ErrorActionPreference = 'Stop'
# A dedicated SSH config avoids inheriting the existing reverse proxy forward.
# ClearAllForwardings=yes would also disable the requested local Web tunnel.
$identity = Join-Path $env:USERPROFILE '.ssh/A6000/kxliu_ed25519'
$certificate = Join-Path $env:USERPROFILE '.ssh/A6000/kxliu_ed25519-cert.pub'
Write-Host "Open http://127.0.0.1:$Port ; keep this terminal running."
while ($true) {
  & ssh -F NUL -o BatchMode=yes -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
    -i $identity -o "CertificateFile=$certificate" `
    -L "127.0.0.1:${Port}:127.0.0.1:${Port}" -N kxliu@100.111.100.200
  Write-Host 'SSH disconnected. Reconnecting in 3 seconds; Ctrl+C stops the tunnel.'
  Start-Sleep -Seconds 3
}
