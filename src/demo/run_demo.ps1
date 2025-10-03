param(
    [string]$Db = "demo.db"
)

raglite init-db --db $Db
raglite ingest --db $Db --path (Join-Path $PSScriptRoot 'mini_corpus') --embed-model debug
raglite query --db $Db --text "quick start guide" --k 3
raglite serve --db $Db --host 127.0.0.1 --port 8000
