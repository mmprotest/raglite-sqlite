Param(
    [string]$Python = "python"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvPath = Join-Path $RepoRoot ".venv"
$DbPath = Join-Path $RepoRoot "demo/demo.db"
$CorpusDir = Join-Path $RepoRoot "demo/mini_corpus"

if (-not (Test-Path $VenvPath)) {
    Write-Host "[raglite-demo] Creating virtual environment at $VenvPath"
    & $Python -m venv $VenvPath
}

$Activate = Join-Path $VenvPath "Scripts/Activate.ps1"
. $Activate

Set-Location $RepoRoot
pip install --upgrade pip
pip install -e .[server]

if (Test-Path $DbPath) {
    Remove-Item $DbPath
}

raglite init-db --db $DbPath
raglite ingest --db $DbPath --path $CorpusDir --embed-model debug --strategy fixed
raglite query --db $DbPath --text "quick start guide" --k 5 --alpha 0.6 --embed-model debug

Write-Host ""
Write-Host "[raglite-demo] Database ready at $DbPath"
Write-Host "[raglite-demo] Starting server on http://127.0.0.1:8080"
Write-Host "[raglite-demo] Try:"
Write-Host "  curl -s http://127.0.0.1:8080/health"
Write-Host "  curl -s -X POST http://127.0.0.1:8080/query `\
    -H 'Content-Type: application/json' `\
    -d '{\"text\": \"backup schedule\", \"k\": 3}'"

try {
    raglite serve --db $DbPath --host 127.0.0.1 --port 8080 --embed-model debug
}
finally {
    Write-Host "[raglite-demo] Shutting down server"
}
