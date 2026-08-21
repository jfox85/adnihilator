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
python_command="$(command -v python3.11 || command -v python3)"

if [[ "$(uname -s)" == "Darwin" ]] &&
  [[ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
  native_prefix=(/usr/bin/arch -arm64)
  runtime_arch="arm64"
else
  native_prefix=()
  runtime_arch="$(uname -m)"
fi

bootstrap_fingerprint="$dependency_hash:$runtime_arch"

if [[ -x .venv/bin/python && -f "$bootstrap_marker" ]] &&
  [[ "$(<"$bootstrap_marker")" == "$bootstrap_fingerprint" ]]; then
  exit 0
fi

"${native_prefix[@]}" uv venv --clear --python "$python_command" .venv
"${native_prefix[@]}" uv pip install --python .venv/bin/python -e '.[dev]'
"$project_dir/.devx/python.sh" -c 'import fastapi, pydantic_core, sqlalchemy, uvicorn, web.app'
printf '%s\n' "$bootstrap_fingerprint" > "$bootstrap_marker"
