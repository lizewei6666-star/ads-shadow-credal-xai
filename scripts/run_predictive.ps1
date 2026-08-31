param(
    [ValidateSet("single", "worker", "aggregate")]
    [string]$Mode = "single",
    [string]$WorkerId = "local",
    [string]$Partitions = "",
    [string]$ResumeRunDir = "",
    [int]$Workers = 4,
    [int]$Pending = 8,
    [ValidateSet("full", "smoke")]
    [string]$Profile = "full"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source

$env:ADS_XAI_EXCEL_PATH = (Resolve-Path (Join-Path $Root "data\ads_crashes.xlsx")).Path
$env:ADS_XAI_BASE_OUTPUT_DIR = Join-Path $Root "outputs"
$env:ADS_XAI_DISTRIBUTED_MODE = $Mode
$env:ADS_XAI_DISTRIBUTED_WORKER_ID = $WorkerId
$env:ADS_XAI_DISTRIBUTED_PARTITIONS = $Partitions
$env:ADS_XAI_RESUME_RUN_DIR = $ResumeRunDir
$env:ADS_XAI_SHADOW_PARALLEL_WORKERS = [string]$Workers
$env:ADS_XAI_SHADOW_MAX_PENDING_TASKS = [string]$Pending
$env:ADS_XAI_RUN_PROFILE = $Profile

& $Python (Join-Path $Root "analysis\run_predictive_credal.py")
exit $LASTEXITCODE

