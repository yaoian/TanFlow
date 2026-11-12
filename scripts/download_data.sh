#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATA_DIR="$ROOT/data"
mkdir -p "$DATA_DIR"

ZENODO_DEFAULT="https://zenodo.org/records/24163449/files/tanflow_benchmarks.tar.gz?download=1"
URL="${1:-$ZENODO_DEFAULT}"

echo "TanFlow Benchmark Datasets Downloader (DOI: 10.5281/zenodo.24163449)"

echo "Downloading from: $URL ..."
curl -fL --progress-bar "$URL" -o "$DATA_DIR/tanflow_benchmarks.tar.gz"

echo "Extracting benchmarks to $DATA_DIR/ ..."
tar -xzvf "$DATA_DIR/tanflow_benchmarks.tar.gz" -C "$DATA_DIR/"
rm -f "$DATA_DIR/tanflow_benchmarks.tar.gz"

echo "Done! Datasets successfully installed to $DATA_DIR/"
echo "Available benchmarks:"
ls -lh "$DATA_DIR"/*.npz
