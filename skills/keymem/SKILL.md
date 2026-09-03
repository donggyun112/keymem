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
- A hit is not loaded content: `read_key(key_id, query, namespace)` →
  `read_memory(memory_id, via_key_id, namespace)`. Traversal may reinforce the selected path and
  learn aliases.
- `<keymem-surfaced>` blocks (the UserPromptSubmit hook) are previews. Load the exact record with
  `read_memory`, apply its `validity` status, and ignore it when irrelevant.

## Freshness and confirmation

- `read_memory` retrieves a fact and may reinforce the key path; it does **not** confirm or deepen
  the memory, verify current truth, or refresh it.
- `fresh` may be used normally. Qualify `aging` facts when currentness matters.
- Never assert a `stale` fact as current. Verify it externally or ask the user.
- Call `confirm_memory` only after an explicit current user assertion, an authoritative current
  source, or direct observation.
- Never call `confirm_memory` merely because `read_memory` returned the content successfully.
- Changed fact → `correct`. Junk fact → `forget`. Wrong key → `dismiss`.

## Remember (close the turn)

- Before ending each reply, silently save any durable name, preference, decision, correction,
  project fact, or goal revealed in the turn.
- `remember(content, keys)` with 3–6 keys. Each key is ONE atomic concept of 1–2 words
  (`"recall"`, `"적중률"` — never `"recall 적중률 개선"`). Phrase keys become unreachable singletons.
- Keys MUST span both Korean and English, plus colloquial variants. Reuse the canonical concepts
  `recall` returned instead of minting near-duplicates.
- Act on `hints` in the response (`near_keys`, `language_note`).
- Use `correct` — never a second `remember` — when a stored fact changes. `forget` to drop one.
- `namespace` groups memories by project/context; pass the active one consistently.
- A turn that surfaced a durable fact but saved nothing is a bug.

## Dismiss (correct a wrong surface)

- When `recall` surfaces a memory that is *fine as a fact* but had no business appearing for this
  query, call `dismiss(memory_id, key_id)` with the key it arrived under.
- A mis-hit left undismissed reinforces the wrong path.
- It weakens that one key→memory pairing and nothing else — the content, the other keys, and the
  memory's reachability are untouched, and the link is floored rather than severed.
- Wrong *fact* → `correct`. Junk fact → `forget`. Wrong *key* → `dismiss`.

## Silence

Never mention memory lookup or saving to the user. Act as if you simply know things.
