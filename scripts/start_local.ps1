$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$ollamaExe = if ($ollamaCommand) { $ollamaCommand.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
if (-not (Test-Path -LiteralPath $ollamaExe)) {
    throw 'Ollama is not installed. Install it from https://ollama.com/download/windows'
}
try {
    $null = Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 3
} catch {
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden
    $ready = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        try {
            $null = Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2
            $ready = $true
            break
        } catch { }
    }
    if (-not $ready) { throw 'Ollama did not start on port 11434.' }
}
$models = (Invoke-RestMethod 'http://127.0.0.1:11434/api/tags').models.name
if ($models -notcontains 'qwen3:4b-instruct') {
    & $ollamaExe pull 'qwen3:4b-instruct'
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
}
& $ollamaExe create 'rag-support-qwen3' -f (Join-Path $projectDir 'Modelfile')
if ($LASTEXITCODE -ne 0) { throw 'Local model configuration failed.' }
$envFile = Join-Path $projectDir '.env'
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $projectDir '.env.example') -Destination $envFile
}
Write-Output 'Local model ready: rag-support-qwen3. API: http://127.0.0.1:11434/v1'
