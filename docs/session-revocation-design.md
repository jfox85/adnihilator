# Revocable admin sessions

Owner decision D7 selects planning server-side session revocation. This proposal is a design, not an implemented authentication change. Auth PR #4 remains the immediate redirect/Unicode/session-format fix. Current cookies are stateless: browser logout deletes that browser's cookie, while a copied cookie remains usable until expiry or a password change.

## Proposed behavior

A successful form login issues a cryptographically random, opaque session token with at least 256 bits of entropy. The browser receives it in the existing HttpOnly, SameSite=Lax cookie; preserve Secure on HTTPS and the current 30-day maximum age. Persist only its SHA-256 digest, creation/expiry time, admin identity, and a credential-generation fingerprint. Never persist or log the raw cookie. Random high-entropy token digests do not need password-style slow hashing.

Authentication requires an unexpired row and a matching fingerprint of the current configured admin credentials. Derive that fingerprint using a dedicated server session secret and HMAC, with unambiguous length-prefixed UTF-8 identity/password encoding. Changing either admin credential or the session secret invalidates every existing row without retaining the password. Preserve PR #4's UTF-8 comparisons. A missing session secret must fail startup after this feature is explicitly enabled, rather than silently creating a new secret on every process start.

Logout revokes the presented row and clears the cookie using identical path/security attributes. A copied token then fails on its next request. Make logout a POST with origin/CSRF validation; a GET must not mutate server state. Repeating logout is harmless. Revoking a session cannot cancel an already-authorized in-flight request; document this practical boundary.

An authenticated, CSRF-protected “Sign out all sessions” action revokes all current rows and clears the caller's cookie. Require fresh admin credentials to avoid a stolen unattended browser silently controlling other sessions. Keep HTTP Basic authentication as the existing explicit alternative: revoking cookies does not revoke knowledge of the admin password. Password rotation remains necessary for leaked Basic credentials.

## Storage and concurrency

Add an `admin_sessions` table using the existing SQLAlchemy startup schema path. It contains a unique token digest primary key, admin identity, credential fingerprint, created_at and expires_at (UTC). Index expiry for bounded cleanup. Logout deletes by exact digest; sign-out-all deletes the admin's rows in one transaction. Concurrent authentication may complete if it read a valid row before revocation committed; subsequent requests fail. No episode, podcast or audio data changes.

Cleanup expired rows in a bounded maintenance operation, not an unbounded delete on every authenticated request. Put a documented limit on active sessions per admin and revoke the oldest when issuing beyond that limit. Multiple web processes must share the same database and session secret. Avoid per-process in-memory revocation lists.

## Compatibility and rollout

Deploy schema support first, then enable the new cookie format at a deliberate cutover. Reject legacy signed stateless cookies after cutover: accepting them indefinitely would preserve the exact copied-cookie bug. Existing admins must log in again once. The dashboard should explain the planned sign-out before rollout. A rollback that re-enables legacy cookie acceptance also restores the old risk; use a new credential/session-secret generation and require a fresh login on any rollback.

Back up the database under the normal owner procedure, apply the additive schema and shared secret securely, deploy all web processes together, restart the web service, and verify login/logout with synthetic or owner-controlled sessions. Never print secrets or cookies in verification logs. No worker restart, paid job, episode requeue or R2 change is required. Production deployment is separate from accepting this design; the current PR contains no migration or runtime code.

## Required implementation verification

- Isolated real database tests prove login, expiry, wrong/unknown token, malformed/non-ASCII input, password/username/secret rotation and cookie flags.
- Two independent clients sharing a copied token prove that logout invalidates the second client. Fresh distinct sessions survive single-session logout; sign-out-all invalidates all. Basic auth retains its documented behavior.
- Real concurrent connections exercise login/revocation races and repeat logout; process restart preserves revocation and valid sessions.
- CSRF/origin checks reject cross-site logout/sign-out-all without weakening normal keyboard navigation, proxies or HTTPS cookie behavior.
- Test legacy-cookie rejection/cutover, additive schema idempotence, cleanup bounds and row limits. Assert raw tokens/passwords are absent from stored rows and captured logs.
- Run both independent code reviewers and a fresh native UI reviewer on the implementation, plus the local full suite. There is currently no repository CI configuration; verify remote checks again before rollout.

No implementation or rollout is claimed by this planning document. Follow-up implementation should use this behavior contract and the existing owner review process.
