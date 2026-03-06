#!/bin/bash
# Launch MLflow UI backed by Cloud SQL via the Auth Proxy.
#
# Usage:
#   ./scripts/mlflow-ui.sh          # default port 5000
#   ./scripts/mlflow-ui.sh 8080     # custom port
#
# Prerequisites:
#   - gcloud auth application-default login  (one-time, for non-GCE machines)
#   - cloud-sql-proxy binary on PATH or in current directory

set -euo pipefail

MLFLOW_PORT="${1:-5000}"
CONNECTION="jomof-sandbox:us-central1:mlflow-db"
PG_URI="postgresql+psycopg2://postgres:mlflow-kotogram-2026@localhost:5432/mlflow"

# Find cloud-sql-proxy
if command -v cloud-sql-proxy &>/dev/null; then
    PROXY=cloud-sql-proxy
elif [ -f ./cloud-sql-proxy ]; then
    PROXY=./cloud-sql-proxy
else
    echo "cloud-sql-proxy not found. Installing..."
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) URL="https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.darwin.arm64" ;;
        Darwin-x86_64) URL="https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.darwin.amd64" ;;
        Linux-x86_64) URL="https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.2/cloud-sql-proxy.linux.amd64" ;;
        *) echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac
    curl -sSL -o cloud-sql-proxy "$URL"
    chmod +x cloud-sql-proxy
    PROXY=./cloud-sql-proxy
fi

cleanup() {
    if [ -n "${PROXY_PID:-}" ]; then
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting Cloud SQL Auth Proxy..."
$PROXY "$CONNECTION" &
PROXY_PID=$!

# Wait for proxy to be ready
for i in $(seq 1 10); do
    if nc -z localhost 5432 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! nc -z localhost 5432 2>/dev/null; then
    echo "Error: proxy failed to start"
    exit 1
fi

echo "Proxy ready. Starting MLflow UI on http://localhost:${MLFLOW_PORT}"
mlflow ui --backend-store-uri "$PG_URI" --port "$MLFLOW_PORT"
