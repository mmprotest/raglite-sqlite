#!/usr/bin/env bash
set -euo pipefail
HOST=${1:-http://127.0.0.1:8000}
for i in {1..5}; do
  curl -s -X POST "$HOST/query" -H 'Content-Type: application/json' -d '{"text": "quick start guide", "k": 3}' >/dev/null
  echo "Completed query $i"
  sleep 1
done
