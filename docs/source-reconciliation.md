# Source reconciliation provenance

This branch reconciles the production checkout's unmerged source with `main` (`834b9c1a0e3de4fabc016fc2bb4fc16d6d57dc70`). It preserves the transcript search, reader, scoped read-token and MCP feature from `f9ee33c825064eeb3f2af9df396352f7b48666d1`, including its migration/import tools and recovery tests.

The intended dirty source adds the google-genai SDK migration, shared model/thinking helpers, client usage accounting, SDK configuration/tests, upload compatibility helper/tests and worker configured-secret error masking. The three guarded root-level manual experiment scripts are retained as source only. They make paid calls when explicitly invoked and are not execution instructions or test evidence.

The original checkout remains untouched. Its private `adnihilator.toml`, credential-bearing worker runner, environment, databases and media were not copied. The public tracked configuration here starts from main and applies only PR7's reviewed SDK3 model/thinking defaults; it does not copy the production private configuration.

## Existing fix overlap

PR7 (`db0eb557aaf47a62a8e01ef7eb71637f12db312d`) independently contains the SDK migration plus additional detection/provider safety changes. This reconciliation retains the production SDK subset and does not absorb the whole detection change. The SDK configuration and dependency files agree with PR7; Gemini/worker implementation chunks overlap and require deliberate composition with PR7's final safety behavior. Do not overwrite PR7 with these earlier production versions.

This feature also intersects the existing auth, feed, worker API, UI and diagnostic batches. Their final fixes remain dependencies for the eventual combined release. Source reconciliation does not approve those blocked dependencies or replace their reviews. Deploy only after root review of the final composition, migrations, CI and outstanding owner decisions.

## Validation boundary

Tests use scratch SQLite, synthetic transcripts and mocked providers. External sockets and DNS are denied. No production worker, paid model request, production migration/import, deployment or restart is performed. The independently hunted feature scope is transcript indexing/search, scoped reads, transcript jobs/recovery, MCP protocol/auth/input handling and transcript UI; confirmation precedes fixes.
