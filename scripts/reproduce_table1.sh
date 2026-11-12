#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -P -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RESULTS_DIR="$ROOT/results"
DEFAULT_OUTPUT="$RESULTS_DIR/table1"
PYTHON="${PYTHON:-python3}"
DEVICE="cpu"
BATCH_SIZE=256
SEED=0
DATA=()
OUTPUT=""
OUTPUT_SET=0
DEMO=0

usage() {
    cat <<'EOF'
Usage: bash scripts/reproduce_table1.sh [OPTIONS]

Train and evaluate TanFlow-Full under the Table 1 evaluation protocol.

Options:
  --demo                 Run a quick smoke test on synthetic data.
  --data DATASET=PATH   Path to normalized NPZ dataset (atc, indoor, sdd, edinburgh,
                         zara1, zara2). Relative paths are resolved from the
                         current working directory. Repeat for each dataset.
  --output DIRECTORY    Output directory for checkpoints and logs. Defaults to
                         results/table1 under the repository root. Relative
                         paths are resolved from the current working directory.
                         Output inside the repository is restricted to results/;
                         paths outside the repository are also accepted. The
                         directory must be empty.
  --python EXECUTABLE   Python executable (default: python3 or $PYTHON).
  --device DEVICE       Torch compute device (e.g., cpu, cuda:0).
  --batch-size N        Batch size (default: 256; demo: 8).
  --seed N              Random seed: 0, 1, or 3 (default: 0).
  -h, --help            Show this help message.

Formal training runs execute 30,000 updates (60,000 on SDD) across gap sizes
M in {5, 13, 21} with validation every 500 steps. Dataset files must follow
the normalized format described in scripts/release_run.py.
EOF
}

require_value() {
    (($# >= 2)) || { printf 'Error: %s requires a value.\n' "$1" >&2; exit 2; }
    [[ -n "$2" ]] || { printf 'Error: %s requires a non-empty value.\n' "$1" >&2; exit 2; }
}

while (($#)); do
    case "$1" in
        --demo) DEMO=1; shift ;;
        --data) require_value "$@"; DATA+=("$2"); shift 2 ;;
        --output) require_value "$@"; OUTPUT="$2"; OUTPUT_SET=1; shift 2 ;;
        --python) require_value "$@"; PYTHON="$2"; shift 2 ;;
        --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
        --batch-size) require_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
        --seed) require_value "$@"; SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$SEED" =~ ^(0|1|3)$ ]] || { printf 'Error: seed must be 0, 1, or 3.\n' >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { printf 'Error: invalid batch size.\n' >&2; exit 2; }
if [[ "$DEMO" -eq 0 && "$BATCH_SIZE" -ne 256 ]]; then
    printf 'Error: formal Table 1 runs require --batch-size 256.\n' >&2; exit 2
fi

if ((OUTPUT_SET)); then
    [[ -n "$OUTPUT" ]] || { printf 'Error: --output requires a non-empty value.\n' >&2; exit 2; }
else
    OUTPUT="$DEFAULT_OUTPUT"
fi

# Normalize an absolute path lexically (no symlink resolution).
normalize_path() {
    local path="$1" part result=""
    local -a parts=()
    IFS=/ read -r -a parts <<< "$path"
    for part in "${parts[@]}"; do
        case "$part" in
            ''|.) ;;
            ..) result="${result%/*}" ;;
            *) result="$result/$part" ;;
        esac
    done
    printf '%s\n' "${result:-/}"
}

# Return an absolute path without creating anything. Existing path components
# are resolved physically (following symlinks); missing trailing components are
# normalized lexically.
resolve_path() {
    local path="$1" suffix=""
    [[ "$path" == /* ]] || path="$PWD/$path"
    while [[ ! -e "$path" ]]; do
        suffix="/${path##*/}$suffix"
        path="${path%/*}"
        [[ -n "$path" ]] || { path="/"; break; }
    done
    local base
    base="$(cd -P -- "$path" && pwd -P)" || return 1
    normalize_path "$base$suffix"
}

# Resolve a dataset file against the caller's working directory. The target must
# be an existing readable regular file; failures name the original argument and
# show how it was resolved.
resolve_data_path() {
    local input="$1" resolved=""
    if [[ -z "$input" ]]; then
        printf 'Error: dataset path is empty.\n' >&2
        return 1
    fi
    if command -v readlink >/dev/null 2>&1; then
        resolved="$(readlink -f -- "$input" 2>/dev/null || true)"
    fi
    if [[ -z "$resolved" ]] && command -v realpath >/dev/null 2>&1; then
        resolved="$(realpath -- "$input" 2>/dev/null || true)"
    fi
    if [[ -z "$resolved" ]]; then
        # Last-resort fallback for systems without readlink -f or realpath.
        if [[ "$input" == /* ]]; then
            resolved="$input"
        else
            resolved="$PWD/$input"
        fi
    fi
    if [[ -z "$resolved" || ! -e "$resolved" ]]; then
        printf 'Error: dataset file does not exist: %s\n' "$input" >&2
        [[ "$input" == /* ]] || printf '       looked relative to: %s\n' "$PWD" >&2
        [[ -z "$resolved" || "$resolved" == "$input" ]] || \
            printf '       resolved path: %s\n' "$resolved" >&2
        return 1
    fi
    if [[ ! -f "$resolved" ]]; then
        printf 'Error: dataset path is not a regular file: %s\n' "$input" >&2
        return 1
    fi
    if [[ ! -r "$resolved" ]]; then
        printf 'Error: dataset file is not readable: %s\n' "$resolved" >&2
        return 1
    fi
    printf '%s\n' "$resolved"
}

# Repository-internal output is allowed only below results/; output outside the
# repository remains unrestricted.
require_output_under_results() {
    case "$1" in
        "$RESULTS_DIR"|"$RESULTS_DIR"/*) ;;
        "$ROOT"|"$ROOT"/*)
            printf 'Error: output inside the repository must be under %s: %s\n' \
                "$RESULTS_DIR" "$1" >&2
            return 2 ;;
    esac
    return 0
}

resolved_output="$(resolve_path "$OUTPUT")" || {
    printf 'Error: cannot resolve output directory: %s\n' "$OUTPUT" >&2; exit 2
}
OUTPUT="$resolved_output"
require_output_under_results "$OUTPUT" || exit 2
mkdir -p -- "$OUTPUT" || {
    printf 'Error: cannot create output directory: %s\n' "$OUTPUT" >&2; exit 2
}
resolved_output="$(cd -P -- "$OUTPUT" && pwd -P)" || {
    printf 'Error: cannot access output directory: %s\n' "$OUTPUT" >&2; exit 2
}
OUTPUT="$resolved_output"
require_output_under_results "$OUTPUT" || exit 2
[[ -z "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    printf 'Error: output directory must be empty: %s\n' "$OUTPUT" >&2; exit 2
}

# Resolve --data paths before changing directories so relative paths keep the
# meaning they had in the caller's working directory.
if [[ "$DEMO" -eq 1 ]]; then
    ((${#DATA[@]} == 0)) || {
        printf 'Error: --demo and --data are mutually exclusive.\n' >&2; exit 2
    }
else
    ((${#DATA[@]})) || { printf 'Error: provide --data or select --demo.\n' >&2; exit 2; }
    resolved_data=()
    for binding in "${DATA[@]}"; do
        dataset="${binding%%=*}"
        path="${binding#*=}"
        if [[ "$dataset" == "$binding" || -z "$dataset" || -z "$path" ]]; then
            printf 'Error: expected DATASET=PATH, got: %s\n' "$binding" >&2
            exit 2
        fi
        resolved_path="$(resolve_data_path "$path")" || exit 2
        resolved_data+=("$dataset=$resolved_path")
        printf 'Dataset %s: %s\n' "$dataset" "$resolved_path"
    done
    DATA=("${resolved_data[@]}")
fi

export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
try:
    import numpy, scipy, yaml, torch
except ImportError as error:
    raise SystemExit(f"Missing dependency: {error.name}; install requirements-freeze.txt") from error
print(f"Environment: Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
PY

cd "$ROOT"
if [[ "$DEMO" -eq 1 ]]; then
    if [[ "$BATCH_SIZE" -eq 256 ]]; then BATCH_SIZE=8; fi
    "$PYTHON" -m scripts.release_run demo-data --output "$OUTPUT/synthetic_atc.npz"
    DATA=("atc=$OUTPUT/synthetic_atc.npz")
fi

for binding in "${DATA[@]}"; do
    dataset="${binding%%=*}"
    path="${binding#*=}"
    [[ "$dataset" != "$binding" && -f "$path" ]] || {
        printf 'Error: expected readable DATASET=PATH, got %s\n' "$binding" >&2; exit 2
    }
    if [[ "$DEMO" -eq 1 ]]; then
        gaps=(13); steps=(4); evaluation_period=2
    else
        gaps=(5 13 21); steps=(30000)
        [[ "$dataset" == "sdd" ]] && steps=(60000)
        evaluation_period=500
    fi
    for gap in "${gaps[@]}"; do
        run="$OUTPUT/${dataset}_mid${gap}_seed${SEED}"
        common=(--data "$path" --dataset "$dataset" --arm TanFlow-Full --gap "$gap"
                --seed "$SEED" --device "$DEVICE" --batch-size "$BATCH_SIZE"
                --eval-every "$evaluation_period")
        evaluate=(--data "$path" --checkpoint "$run/best.pt"
                  --output "${run}_evaluation" --device "$DEVICE" --gaps "$gap")
        if [[ "$DEMO" -eq 1 ]]; then
            "$PYTHON" -m scripts.release_run train --output "$run" --steps "${steps[0]}" --smoke "${common[@]}"
            "$PYTHON" -m scripts.release_run evaluate --allow-smoke "${evaluate[@]}"
        else
            "$PYTHON" -m scripts.release_run train --output "$run" "${common[@]}"
            "$PYTHON" -m scripts.release_run evaluate "${evaluate[@]}"
        fi
    done
done
if [[ "$DEMO" -eq 1 ]]; then
    printf 'Completed requested runs. Demo outputs are not paper reproductions.\n'
else
    printf 'Completed requested runs.\n'
fi
