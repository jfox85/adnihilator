# Provider retries and outage recovery

Gemini generation makes at most three attempts for one detection decision. Ordinary 429, 500, 502, 503, 504, transport errors and timeouts use bounded exponential delays of 30 and 60 seconds. One malformed JSON/schema response may be reasked, consuming the same three-attempt decision budget. Refusal, blocked or unsuccessful completion metadata is rejected. Permanent credit, authentication and model errors do not retry within the operation. Each upload/status operation has its own three-attempt bound; generation retries reuse the uploaded file. SDK nested HTTP retries are disabled, including resumable-upload error responses. Both installed HTTPX and optional HTTPX2 transport exceptions follow this budget. Explicit missing/expired uploaded File errors fail the episode without declaring an account-wide model outage.

A required Gemini or OpenAI account outage returns the affected episode to a durable provider hold without changing its retry count. All new claims pause. Normal outages automatically become eligible after 300 seconds, doubling on failed recovery to a maximum of 1,800 seconds. `PROVIDER_RECOVERY_BASE_SECONDS` and `PROVIDER_RECOVERY_MAX_SECONDS` may lower these bounds (positive integers, base <= maximum <= 1800). No separate paid health request is made: one ordinary queued episode obtains a recovery lease. Other claims remain paused until a completed episode reports successful use of the affected provider. A successful path that did not call that provider leaves its outage active. Multiple simultaneous provider outages share one episode lease.

Dashboard and queue banners survive restarts, show held episodes and the recovery state, and offer Retry Now. This accelerates eligibility without duplicating an active recovery lease. The authenticated action requires a same-origin form submission and confirms possible provider charges. Generic recovery failures and expired recovery leases retain episode retry budgets. Probes rotate durably through never-probed, then least-recently-probed pending episodes, preferring held work when otherwise tied. A broken audio source cannot monopolize recovery while another eligible episode is pending. Recovery-specific tokens reject stale callbacks; general attempt-token fencing remains deferred.

## Deployment prerequisites and migration

This change depends on the reviewed provider safety/Google SDK changes (PRs 3 and 7), API queue guards (PR 8), and the approved canonical redaction helper from commit `12fca1c87a7228e778dc565ec58f4ae2361114b1` (PR 12). It does not depend on the separate dashboard error-handling PR 11. The source contains the exact approved helper bytes, SHA256 `096e5de9f34ef3a35fd4463dccece121d864a67905bce0fa3cf8698b0336d10a`.

The migration adds only `provider_outages` and `episode_provider_holds` through the existing SQLAlchemy `create_all` startup path. Existing episode columns, status constraints and retry counts are unchanged. Back up the database before the separately approved rollout. A rollback must account for held pending episodes before restoring an older worker; an older worker does not understand the pause protocol.

Deploy the approved reconciled release with workers idle/stopped, preserving private configuration, transcripts and the approved release baseline. Install dependencies using the actual approved worker interpreter, never a development environment that resolves to production. Copy new modules before their callers: `adnihilator/gemini_retry.py`, `web/services/provider_outages.py`, and `web/templates/provider_outage_banner.html`, together with the reviewed `provider_responses.py`, canonical `redaction.py`, and all changed existing files. Compile the complete reconciled release before starting services. No production changes are authorized by this document.

Stage the new web/API before the new worker. The new worker uses `/api/queue/claim-with-provider-recovery` with protocol `provider-outage-v1`; an old server's 404 stops claiming rather than falling back to an unsafe legacy path. Existing healthy legacy API clients remain compatible, but only the new protocol can obtain a recovery lease.

Before loading workers during rollout, establish a durable maintenance pause in the approved target database using the approved server interpreter and a fresh session:

```python
from web.database import init_db, get_session_factory
from web.services.provider_outages import pause_for_rollout
engine = init_db()
with get_session_factory(engine)() as db:
    pause_for_rollout(db, "gemini")
```

This owner rollout pause survives restarts and does not automatically probe queued episodes. Add an OpenAI rollout pause similarly if that provider is affected. After the owner confirms the A1 credit/resumption window and provider configuration, the banner's Resume action permits one ordinary recovery episode. Subsequent normal outages recover automatically with the bounded cooldown; they do not require a new manual acknowledgment. Do not requeue historical failed jobs or issue detector health calls as part of migration. All rollout commands require the separate owner-approved release/deployment action.
