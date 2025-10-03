#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON:-python3}"
DB_PATH="${ROOT_DIR}/demo/demo.db"
CORPUS_DIR="${ROOT_DIR}/demo/mini_corpus"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[raglite-demo] Creating virtual environment at ${VENV_DIR}" >&2
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -e "${ROOT_DIR}.[server]"

rm -f "${DB_PATH}"
raglite init-db --db "${DB_PATH}"
raglite ingest --db "${DB_PATH}" --path "${CORPUS_DIR}" --embed-model debug --strategy fixed
raglite query --db "${DB_PATH}" --text "quick start guide" --k 5 --alpha 0.6 --embed-model debug

echo ""
echo "[raglite-demo] Database ready at ${DB_PATH}" >&2
echo "[raglite-demo] Starting server on http://127.0.0.1:8080" >&2
echo "[raglite-demo] Try:" >&2
echo "  curl -s http://127.0.0.1:8080/health" >&2
echo "  curl -s -X POST http://127.0.0.1:8080/query \\\" >&2
echo "    -H 'Content-Type: application/json' \\\" >&2
echo "    -d '{\"text\": \"backup schedule\", \"k\": 3}'" >&2

trap_exit() {
  echo "[raglite-demo] Shutting down server" >&2
  exit 0
}
trap trap_exit INT TERM

raglite serve --db "${DB_PATH}" --host 127.0.0.1 --port 8080 --embed-model debug
