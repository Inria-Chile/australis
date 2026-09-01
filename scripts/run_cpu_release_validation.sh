#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: run_cpu_release_validation.sh RUN_DIR}"
test -d "$RUN_DIR"
mkdir -p "$RUN_DIR/exports/dist" "$RUN_DIR/logs"

uv run ruff check . >"$RUN_DIR/logs/ruff.log" 2>&1
uv run pytest -q >"$RUN_DIR/logs/pytest.log" 2>&1
uv build --out-dir "$RUN_DIR/exports/dist" >"$RUN_DIR/logs/build.log" 2>&1

wheel=$(find "$RUN_DIR/exports/dist" -maxdepth 1 -name '*.whl' -print -quit)
sdist=$(find "$RUN_DIR/exports/dist" -maxdepth 1 -name '*.tar.gz' -print -quit)
test -n "$wheel"
test -n "$sdist"
uv run python scripts/check_release_readiness.py \
  --root . \
  --archive "$wheel" \
  --archive "$sdist" \
  --output "$RUN_DIR/exports/release_readiness.json" \
  >"$RUN_DIR/logs/release_readiness.log" 2>&1

uv venv "$RUN_DIR/release-venv" --python "$(command -v python3)" \
  >"$RUN_DIR/logs/wheel_smoke_install.log" 2>&1
uv pip install --python "$RUN_DIR/release-venv/bin/python" "$wheel" \
  >>"$RUN_DIR/logs/wheel_smoke_install.log" 2>&1
"$RUN_DIR/release-venv/bin/polarfunc" --help \
  >"$RUN_DIR/logs/wheel_smoke_help.log" 2>&1

find "$RUN_DIR/exports" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"$RUN_DIR/exports/SHA256SUMS"
(cd / && sha256sum -c "$RUN_DIR/exports/SHA256SUMS") \
  >"$RUN_DIR/logs/checksums.log" 2>&1
printf 'status=PASS\n' >"$RUN_DIR/completion.env"
