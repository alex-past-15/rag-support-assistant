$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
& (Join-Path $PSScriptRoot 'start_local.ps1')
$pythonExe = Join-Path $projectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = Join-Path (Split-Path $projectDir -Parent) '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create the Python environment and install requirements.txt first.' }
$env:HF_HUB_OFFLINE = '1'
& $pythonExe -m streamlit run (Join-Path $projectDir 'app.py') --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false
