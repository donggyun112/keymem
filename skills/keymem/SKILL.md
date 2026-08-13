---
name: keymem
description: Use when the conversation reveals or needs durable knowledge about the user, the project, or an earlier decision — the recall/remember protocol for keymem's associative memory (keys, namespaces, corrections, and the passively surfaced <keymem-surfaced> hints).
---

# keymem protocol

## Recall (open the turn)

- Before the first reply, and whenever the topic shifts, call `recall` silently.
- `query` = short noun keywords (`"거주지"`, not `"어디 살아"`); `context` = the raw user
  utterance. Keys match keywords, content matches sentences — the two cues take different paths.
- Split multi-fact questions into several `recall` calls.
- On `{status:"no_match"}`, retry with one of the returned `nearest_keys`, or `browse_keys(namespace)`.
- A hit is not a fact yet: `read_key(key_id, query, namespace)` → `read_memory(memory_id, via_key_id, namespace)`.
  Only the full traversal reinforces the path and learns aliases.
- `<keymem-surfaced>` blocks (the UserPromptSubmit hook) are unconfirmed previews — verify with
  `read_memory` before asserting, ignore when irrelevant.

## Remember (close the turn)

- Before ending every reply, check whether the turn revealed something durable: a name, preference,
  decision, correction, project fact, or goal. If it did, save it silently in the same turn.
- `remember(content, keys)` with 3–6 keys. Each key is ONE atomic concept of 1–2 words
  (`"recall"`, `"적중률"` — never `"recall 적중률 개선"`). Phrase keys become unreachable singletons.
- Keys MUST span both Korean and English, plus colloquial variants. Reuse the canonical concepts
  `recall` returned instead of minting near-duplicates.
- Act on `hints` in the response (`near_keys`, `language_note`).
- Use `correct` — never a second `remember` — when a stored fact changes. `forget` to drop one.
- `namespace` groups memories by project/context; pass the active one consistently.
- A turn that surfaced a durable fact but saved nothing is a bug.

## Silence

Never mention memory lookup or saving to the user. Act as if you simply know things.
