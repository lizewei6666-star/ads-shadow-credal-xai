$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source

$Scripts = @(
    "make_figure1.py",
    "make_figure2.py",
    "make_figure3.py",
    "make_figures4_5.py",
    "make_figure6.py",
    "make_figure7.py"
)

foreach ($Script in $Scripts) {
    Write-Host "Running $Script"
    & $Python (Join-Path $Root "figures\$Script")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Figure reproduction completed."

