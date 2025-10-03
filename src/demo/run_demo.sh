#!/usr/bin/env bash
set -euo pipefail
DB=${1:-demo.db}

raglite init-db --db "$DB"
raglite ingest --db "$DB" --path "$(dirname "$0")/mini_corpus" --embed-model debug
raglite query --db "$DB" --text "quick start guide" --k 3
raglite serve --db "$DB" --host 127.0.0.1 --port 8000
