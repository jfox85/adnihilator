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
Refresh, Retry and Refresh All use the same import boundary. A successful sync
clears the recorded error and updates the last-success timestamp. Failed imports
roll back episode/metadata changes before saving diagnostics. A database outage
can also prevent saving diagnostics; the original safe failure is still returned.
The nullable error column is added to an existing SQLite database at startup;
no existing episode values are rewritten and no separate migration is required.

Episode identity follows this precedence:

1. Nonempty feed GUID/Atom ID, as normalized by feedparser.
2. Nonempty episode page link, as normalized by feedparser.
3. Selected usable audio enclosure URL: `enclosure-sha256:` followed by SHA-256
   of normalized scheme, lowercase host, nondefault port and path. Empty path is
   `/`. Userinfo, query and fragment are excluded.

The third rule keeps query-based signature/tracking-token rotation stable,
including across a restart. A different path, host, nondefault port or scheme
is a different fallback identity. Reusing the exact path for different episodes
without GUIDs/page links collides; changing the path for the same episode creates
a new identity. Feed authors should provide stable unique GUIDs. There is no
heuristic title/date matching or automatic identity migration. Existing GUID rows
are not silently backfilled or modified, including their missing or expired audio
URLs. A signed enclosure that expires still needs a separate explicit recovery
policy; this change does not repair it by overwriting durable rows.

Enclosures prefer explicit `audio/*` MIME types with usable HTTP(S) URLs. When
MIME is absent, `application/octet-stream`, or `binary/octet-stream`, a recognized
audio pathname extension (case-insensitive, before query/fragment) is accepted.
Examples include `.mp3`, `.m4a`, `.aac`, `.ogg`, `.opus`, `.wav` and `.flac`; the
complete list is `_AUDIO_EXTENSIONS` in `web/services/audio.py`. Explicit image,
video, PDF or other non-audio MIME types remain rejected even with an audio
extension. Extension fallback is an admission rule, not verification of bytes.

Ordinary Queue rejects completed episodes, including a completion that wins
between its read and conditional update. It preserves outputs, retry counts and
diagnostics. No Reprocess UI or endpoint is introduced (D10); an explicit workflow
can be designed separately. D9 attempt-token fencing remains deferred: these
status/lease checks do not identify successive processing attempts.
