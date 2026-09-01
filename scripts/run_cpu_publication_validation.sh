#!/usr/bin/env bash
set -euo pipefail

: "${POLARFUNC_DATA_ROOT:?POLARFUNC_DATA_ROOT is required}"
: "${POLARFUNC_HISTORICAL_ROOT:?POLARFUNC_HISTORICAL_ROOT is required}"
: "${POLARFUNC_OUTPUT_ROOT:?POLARFUNC_OUTPUT_ROOT is required}"

RUN_DIR="${1:?usage: run_cpu_publication_validation.sh RUN_DIR}"
test -d "$RUN_DIR"
mkdir -p "$RUN_DIR/exports" "$RUN_DIR/logs"

uv run ruff check . >"$RUN_DIR/logs/ruff.log" 2>&1
uv run pytest -q >"$RUN_DIR/logs/pytest.log" 2>&1
uv build --out-dir "$RUN_DIR/exports/dist" >"$RUN_DIR/logs/build.log" 2>&1

uv run polarfunc zenodo-manifest 14181291 \
  --output "$RUN_DIR/exports/ace_external_artifacts.json" \
  >"$RUN_DIR/logs/zenodo_manifest.log" 2>&1

uv run polarfunc verify-manifest "$RUN_DIR/exports/ace_external_artifacts.json" \
  --root "$POLARFUNC_DATA_ROOT" \
  --workers 4 \
  --output "$RUN_DIR/exports/ace_download_verification.json" \
  >"$RUN_DIR/logs/ace_verification.log" 2>&1

uv run python scripts/validate_unigene_master.py \
  "$POLARFUNC_HISTORICAL_ROOT/artifacts/master/unigene_master" \
  --checksums \
  --output "$RUN_DIR/exports/unigene_master_validation.json" \
  >"$RUN_DIR/logs/unigene_master.log" 2>&1

if test -d "$POLARFUNC_HISTORICAL_ROOT/artifacts/structure_branch/esm3_common"; then
  uv run polarfunc bucket-index \
    --root "$POLARFUNC_HISTORICAL_ROOT/artifacts/structure_branch/esm3_common" \
    --pattern '*.pdb' \
    --buckets 256 \
    --output "$RUN_DIR/exports/esm3_structure_bucket_index.jsonl" \
    >"$RUN_DIR/logs/structure_index.log" 2>&1
fi

find "$RUN_DIR/exports" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
  >"$RUN_DIR/exports/SHA256SUMS"
printf 'status=PASS\n' >"$RUN_DIR/completion.env"
