<#
.SYNOPSIS
Build dist\QualityScaler.exe using Python 3.11 and the root PyInstaller spec.

.DESCRIPTION
Runs from any working directory and restores the caller's location on exit.
Uses .venv\Scripts\python.exe; if .venv does not exist, creates it with
py -3.11 -m venv .venv. No administrator rights are required.

By default, installs requirements.txt into .venv before building. -SkipInstall
skips all pip operations, so the caller must have already provisioned .venv
with those requirements. It does not skip resource checks or the build.

Before building, provide Assets\ffmpeg.exe, Assets\exiftool.exe, the app's PNG/ICO
assets, and all seven standard AI-onnx\*_fp16.onnx models listed in the spec.
Additional local *_fp16.onnx models are also bundled.
The spec fails if required local resources or DirectML runtime DLLs are absent.
Produces one windowed executable, without UPX, signing, or an installer.

.EXAMPLE
.\build.ps1

.EXAMPLE
.\build.ps1 -SkipInstall
#>
[CmdletBinding()]
param(
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$CallerLocation = Get-Location
try {
    Set-Location -LiteralPath $PSScriptRoot

    $VenvPath = Join-Path $PSScriptRoot '.venv'
    $PythonPath = Join-Path $VenvPath 'Scripts\python.exe'
    $RequirementsPath = Join-Path $PSScriptRoot 'requirements.txt'
    $SpecPath = Join-Path $PSScriptRoot 'QualityScaler.spec'
    $ArtifactPath = Join-Path $PSScriptRoot 'dist\QualityScaler.exe'

    if (-not (Test-Path -LiteralPath $VenvPath)) {
        & py -3.11 -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 virtual environment creation failed (exit $LASTEXITCODE)."
        }
    }
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Missing $PythonPath. Provision a Python 3.11 .venv before building."
    }

    & $PythonPath -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'
    if ($LASTEXITCODE -ne 0) {
        throw 'The existing .venv must use Python 3.11. No environment was removed.'
    }

    if (-not $SkipInstall) {
        & $PythonPath -m pip install --disable-pip-version-check --no-input -r $RequirementsPath
        if ($LASTEXITCODE -ne 0) {
            throw "Installing requirements.txt failed (exit $LASTEXITCODE)."
        }
    }

    & $PythonPath -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed (exit $LASTEXITCODE). No new artifact is being reported."
    }
    if (-not (Test-Path -LiteralPath $ArtifactPath -PathType Leaf)) {
        throw "PyInstaller returned success but the artifact is missing: $ArtifactPath"
    }

    $Artifact = Get-Item -LiteralPath $ArtifactPath
    $Hash = Get-FileHash -LiteralPath $ArtifactPath -Algorithm SHA256
    Write-Host ("Artifact: {0}" -f $Artifact.FullName)
    Write-Host ("Size: {0:N0} bytes" -f $Artifact.Length)
    Write-Host ("SHA256: {0}" -f $Hash.Hash)
}
finally {
    Set-Location -LiteralPath $CallerLocation.Path
}
