# Remove body embeddings

The user explicitly requested complete removal of body embeddings after comparing
Kiwi BM25, key embeddings and reranking. Key embeddings and the reranker stay.

- [x] Pin writes, corrections, retrieval and legacy migration with a failing regression test.
- [x] Remove `Memory.embedding`, body-vector generation, persistence, similarity gates,
  semantic deduplication/contradiction detection and body-driven automatic key links.
  Keep exact normalized-text deduplication and explicit correction/link APIs.
- [x] Use Kiwi BM25 over memory text and authored key/alias text; preserve key embeddings
  and graph navigation. Rerank candidates without storing body vectors. BM25 candidates
  must not be discarded for lacking a body cosine anchor.
- [x] Load old inline/sidecar graphs without embedding any memory; retain content,
  namespace, history and links, and write only key vectors on the next successful save.
- [x] Update active docs and affected tests, run the suite/build, evaluate the actual new
  retrieval path and inspect a clean package. Do not publish or overwrite the live store.

Work is executed in this chat. Tokenizer and existing-test updates are independently
delegated under dispatching-parallel-agents; the graph implementation stays with root.

Validation: 278/278 tests pass, TypeScript/build pass, real bge-m3-key/reranker
fixture evaluation completes, and a fresh npm-tarball install loads actual Kiwi
and the native reranker while writing key-only vectors. Independent review
findings about pagination and scoped candidate limits have regression coverage.
