$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $PSScriptRoot 'cppcheck'
$msiPath = Join-Path $toolRoot 'cppcheck-2.22.0-x64-Setup.msi'
$extractRoot = Join-Path $toolRoot 'extracted'
$cppcheckExe = Join-Path $extractRoot 'PFiles\Cppcheck\cppcheck.exe'
$expectedSha256 = '3929162bab7e71d1b9b18708ec054fb0188bfc8ea1660979ed95837f5a9f547b'
$downloadUrl = 'https://github.com/cppcheck-opensource/cppcheck/releases/download/2.22.0/cppcheck-2.22.0-x64-Setup.msi'

New-Item -ItemType Directory -Path $toolRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $msiPath)) {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $msiPath -TimeoutSec 120
}
$actualSha256 = (Get-FileHash -LiteralPath $msiPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    throw 'Cppcheck installer checksum mismatch. Do not run this installer.'
}

if (-not (Test-Path -LiteralPath $cppcheckExe)) {
    $process = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/a', $msiPath, '/qn', ('TARGETDIR=' + $extractRoot)) -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "Cppcheck extraction failed with exit code $($process.ExitCode)"
    }
}
if (-not (Test-Path -LiteralPath $cppcheckExe)) {
    throw 'Cppcheck executable was not found after extraction.'
}
Write-Output "Project: $projectRoot"
& $cppcheckExe --version
