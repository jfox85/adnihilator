# AdNihilator DevX

DevX sessions run on the macOS host so native audio dependencies and ffmpeg
remain available. Each session receives an allocated `API` port and a local
Caddy route.

The `service` window starts only the FastAPI web application. It does not start
the worker daemon, read `scripts/run-worker.sh`, copy production data, download
audio or speech models, or load production credentials.

## Isolation

Generated state stays inside the session worktree under `.devx/runtime/`:

- `adnihilator.db` is the session-only SQLite database.
- `web.env` contains randomly generated development-only admin and worker
  credentials and is created with mode `0600`.
- `bootstrap.sha256` records the installed `pyproject.toml` dependency state.

On Apple Silicon, the bootstrap and `python.sh` wrapper force native arm64
execution even when a long-lived tmux server was originally started through
Rosetta. This keeps compiled Python extensions consistent across service,
interactive, and non-interactive commands.

Feed synchronization is disabled with `FEED_SYNC_ENABLED=false`, so starting a
session does not fetch subscribed feeds in the background. The generated worker
key exists only so the web API can initialize safely; no worker is launched.

To inspect the development login locally, open `.devx/runtime/web.env` from the
session worktree. Do not reuse these generated values outside the session.

## Checks

```bash
.devx/bootstrap.sh
.devx/python.sh -m pytest -q
.devx/python.sh -m compileall -q adnihilator web worker tests
curl -fsS "http://127.0.0.1:$API/health"
```
