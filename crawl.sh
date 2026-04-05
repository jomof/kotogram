#!/usr/bin/env bash
set -euo pipefail

while true; do
    scripts/cc extract-sentences --min-impact 0.8
    scripts/curate upsert --allow-insert --sentences .cc/selected-sentences.txt
    sqlite3 data/corpus.db "VACUUM"
    wc -c data/corpus.db
    echo "Sleeping 5 minutes..."
    sleep 300
done
