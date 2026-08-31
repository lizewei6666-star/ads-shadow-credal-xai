param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRun,
    [int]$Workers = 6,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
$Excel = (Resolve-Path (Join-Path $Root "data\ads_crashes.xlsx")).Path
$Source = (Resolve-Path $SourceRun).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $Root "outputs\pag_multiscale_extension_v1"
}

& $Python (Join-Path $Root "analysis\run_pag_structural.py") `
    --excel $Excel `
    --source-run $Source `
    --output-dir $OutputDir `
    --workers $Workers
exit $LASTEXITCODE

