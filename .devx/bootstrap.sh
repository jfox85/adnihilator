#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required to bootstrap AdNihilator" >&2
  exit 1
}

umask 077
mkdir -p .devx/runtime

dependency_hash="$(shasum -a 256 pyproject.toml | awk '{print $1}')"
bootstrap_marker=".devx/runtime/bootstrap.sha256"

if [[ -x .venv/bin/python && -f "$bootstrap_marker" ]] &&
  [[ "$(<"$bootstrap_marker")" == "$dependency_hash" ]]; then
  exit 0
fi

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -c 'import fastapi, sqlalchemy, uvicorn, web.app'
printf '%s\n' "$dependency_hash" > "$bootstrap_marker"
