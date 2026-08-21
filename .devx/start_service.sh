#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

: "${API:?DevX must provide an API port}"

"$project_dir/.devx/bootstrap.sh"

umask 077
runtime_dir="$project_dir/.devx/runtime"
credentials_file="$runtime_dir/web.env"
mkdir -p "$runtime_dir"

if [[ ! -f "$credentials_file" ]]; then
  admin_password="$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(24))')"
  worker_api_key="$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(24))')"
  {
    printf 'ADMIN_USERNAME=%s\n' 'devx-admin'
    printf 'ADMIN_PASSWORD=%s\n' "$admin_password"
    printf 'WORKER_API_KEY=%s\n' "$worker_api_key"
  } > "$credentials_file"
  chmod 600 "$credentials_file"
fi

while IFS='=' read -r key value; do
  case "$key" in
    ADMIN_USERNAME|ADMIN_PASSWORD|WORKER_API_KEY)
      export "$key=$value"
      ;;
  esac
done < "$credentials_file"

export DATABASE_PATH="$runtime_dir/adnihilator.db"
export FEED_SYNC_ENABLED=false
export R2_PUBLIC_URL=""

exec .venv/bin/python -m uvicorn web.app:app \
  --host 127.0.0.1 \
  --port "$API" \
  --reload
