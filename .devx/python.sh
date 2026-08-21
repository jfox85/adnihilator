#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="$project_dir/.venv/bin/python"

if [[ ! -x "$python_command" ]]; then
  echo "AdNihilator virtual environment is not bootstrapped" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]] &&
  [[ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
  exec /usr/bin/arch -arm64 "$python_command" "$@"
fi

exec "$python_command" "$@"
