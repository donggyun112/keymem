# Default Direct-Hydrate Recall Implementation Plan

1. Add a failing MCP contract test requiring default `recall` to return
   `{status, query, namespace, keys, memories}` with one passive Top-1 memory.
2. Reuse query-aware `readKey(..., limit: 1)` internally and return the memory without mutation.
3. Add a failing test requiring `connected_keys` concept/ID references on that memory.
4. Populate `connected_keys` from the existing memory→key graph index.
5. Keep `read_key → read_memory` as the explicit deeper-hop and reinforcement path.
6. Remove the shadow feature flag, logger, JSONL storage, and opt-in documentation.
7. Update MCP instructions, the bundled skill, and README to describe the default core behavior.
8. Run focused tests, the full test suite, and the build.
