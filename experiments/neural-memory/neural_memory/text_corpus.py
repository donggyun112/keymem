from __future__ import annotations


CONTEXT_TEXTS = {
    "train": (
        ("planning a trip", "vacation logistics", "organizing travel"),
        ("writing software", "programming work", "building an application"),
        ("choosing meals", "food preferences", "deciding what to eat"),
        ("listening to music", "music preferences", "choosing something to hear"),
    ),
    "eval": (
        ("preparing for a journey", "making holiday arrangements"),
        ("developing computer code", "working on a software project"),
        ("selecting something for dinner", "picking a dish"),
        ("deciding what songs to play", "selecting audio entertainment"),
    ),
}


CUE_TEXTS = {
    "train": (
        (
            "This has been my consistent personal preference for years.",
            "I use this as my normal default every time.",
            "I changed my standard choice and will use this from now on.",
            "This is a stable part of how I usually do things.",
        ),
        (
            "I happened to see this once today.",
            "Someone else briefly mentioned this in passing.",
            "A random example paired these things in an article.",
            "I tried this once, but it was only temporary.",
        ),
    ),
    "eval": (
        (
            "This remains my regular choice over the long term.",
            "Going forward, this is the option I intend to use by default.",
            "This reflects an enduring preference of mine.",
        ),
        (
            "This was merely something I noticed for a moment.",
            "It came up in somebody else's short anecdote.",
            "This was a one-off occurrence rather than my usual choice.",
        ),
    ),
}


def all_texts() -> tuple[list[str], dict[tuple[str, str], tuple[int, ...]]]:
    """Flatten the corpus and return indices for rebuilding tensor banks."""

    texts: list[str] = []
    groups: dict[tuple[str, str], tuple[int, ...]] = {}
    for split, topics in CONTEXT_TEXTS.items():
        start = len(texts)
        for variants in topics:
            texts.extend(variants)
        groups[(split, "context")] = tuple(range(start, len(texts)))
    for split, classes in CUE_TEXTS.items():
        start = len(texts)
        for variants in classes:
            texts.extend(variants)
        groups[(split, "cue")] = tuple(range(start, len(texts)))
    return texts, groups
