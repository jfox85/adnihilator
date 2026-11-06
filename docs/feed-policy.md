# Subscription, identity and queue policies

Feed subscriptions accept HTTP and HTTPS URLs with a host. Private, loopback,
link-local and LAN hosts are deliberately allowed by owner decision D11. This
is not an SSRF host-blocking policy. Every HTTP redirect must also resolve to
HTTP(S); local paths, file, FTP, data and other schemes are rejected. Feedparser
still resolves ordinary relative links and enclosures against the fetched feed.
Do not expose subscription creation to untrusted users without revisiting D11.

A failed initial fetch/import retains the subscription, records a safe persistent
`last_sync_error`, and displays **Subscription needs attention** plus **Retry**
on the dashboard when no title was fetched. Existing titles remain visible.
Dashboard and episode views show an actionable explanation and Retry for the
same recorded failure. Browser form validation stays in the dashboard and keeps
the attempted URL editable; API clients still receive structured 422 errors.
Detailed safe diagnostics remain stored unchanged; presentation omits internal
exception classes. Connection failures are distinct from non-feed content.
Refresh, Retry and Refresh All use the same import boundary. A successful sync
clears the recorded error and updates the last-success timestamp in the same
commit, including an error saved by a competing refresh after its old snapshot. Failed imports
roll back episode/metadata changes before saving diagnostics. A database outage
can also prevent saving diagnostics; the original safe failure is still returned.
The nullable error column is added to an existing SQLite database at startup;
no existing episode values are rewritten and no separate migration is required.

Episode identity follows this precedence:

1. Nonempty feed GUID/Atom ID, as normalized by feedparser.
2. Nonempty episode page link, as normalized by feedparser.
3. Selected usable audio enclosure URL: `enclosure-sha256:` followed by SHA-256
   of normalized scheme, lowercase host, nondefault port, path and **the entire
   opaque query**. Empty path is `/`. Userinfo and fragment are excluded.

The same complete URL remains stable across refreshes and restarts. Different
identity-bearing queries such as `download.php?episode=1` and `?episode=2` remain
separate. The service does not guess which unknown query parameters are merely
tracking or authentication: discarding all queries silently loses real episodes.
Query-token rotation on a GUID-less entry can therefore add another row. A
publisher should supply a stable GUID or episode page link when its signed audio
URLs rotate. Reusing the same complete URL for different episodes also collides;
there is no title/date heuristic. Existing GUIDs, URLs and output state are not
backfilled or rewritten. Missing/expired saved audio still needs a separate
explicit recovery policy.

Fatal XML syntax errors, including truncated RSS, are rejected before changing
metadata or importing partial entries. Unrecognized XML does not count as a feed
merely because the parser exposes an entry. Recoverable MIME/encoding warnings on
otherwise complete recognized feeds remain supported.

Relative RSS enclosures are resolved against the final fetched HTTP(S) feed URL,
including after redirects. Explicit malformed or unsupported absolute URLs are
not repaired. For Atom, the original enclosure type attribute is retained for
admission: a missing type is distinct from an explicitly declared `text/html`,
even though feedparser defaults both to that label. The same HTTP response is
used for this metadata; there is no second fetch. Gzip/deflate and XML encodings
are handled before reading the source attributes.

Enclosures prefer explicit `audio/*` MIME types with usable HTTP(S) URLs. When
MIME is absent, `application/octet-stream`, or `binary/octet-stream`, a recognized
audio pathname extension (case-insensitive, before query/fragment) is accepted.
Examples include `.mp3`, `.m4a`, `.aac`, `.ogg`, `.opus`, `.wav` and `.flac`; the
complete list is `_AUDIO_EXTENSIONS` in `web/services/audio.py`. Explicit image,
video, PDF or other non-audio MIME types remain rejected even with an audio
extension. Extension fallback is an admission rule, not verification of bytes.

OPML intake uses the same HTTP(S) source validation as direct subscriptions.
Unsupported sources are skipped before persistence and reported separately from
duplicates. The central sync boundary also guards previously stored invalid
sources before opening any parser/file/FTP handler.

Ordinary Queue and Skip reject completed episodes, including a completion that wins
between the read and conditional update. A stale Skip cannot revoke completion
and create an implicit Queue/reprocess path. These actions preserve outputs, retry counts and
diagnostics. No Reprocess UI or endpoint is introduced (D10); an explicit workflow
can be designed separately. D9 attempt-token fencing remains deferred: these
status/lease checks do not identify successive processing attempts.

## Credentials on feed redirects

Feed fetches retain Authorization only within the same origin: scheme, hostname
and effective port. Hostname case and explicit default-port spelling describe
the same origin. A change of scheme, hostname or port removes the complete
Authorization header before the redirected request is opened. HTTP(S) redirects
to private or public destinations remain allowed. Same-origin authenticated
feeds keep working. A publisher that requires authentication at another origin
must provide an appropriate endpoint. Once a chain crosses an origin, the old
Authorization header remains removed on subsequent hops.
