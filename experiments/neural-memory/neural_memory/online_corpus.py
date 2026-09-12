from __future__ import annotations


ONLINE_CONTEXT_TEXTS = {
    "train": (
        ("planning a trip", "vacation logistics", "organizing travel"),
        ("writing software", "programming work", "building an application"),
        ("choosing meals", "food preferences", "deciding what to eat"),
        ("listening to music", "music preferences", "choosing something to hear"),
        ("personal exercise", "fitness training", "planning a workout"),
        ("managing money", "personal budgeting", "organizing my finances"),
        ("household routines", "organizing the home", "planning domestic chores"),
        ("reading books", "literary preferences", "choosing reading material"),
    ),
    "eval": (
        ("preparing for a journey", "making holiday arrangements"),
        ("developing computer code", "working on a software project"),
        ("selecting something for dinner", "picking a dish"),
        ("deciding what songs to play", "selecting audio entertainment"),
        ("deciding on physical activity", "setting up a training session"),
        ("planning how to spend and save", "looking after my household budget"),
        ("taking care of tasks around the house", "arranging everyday home duties"),
        ("selecting a novel to read", "deciding on written material"),
    ),
}


def online_texts() -> tuple[list[str], dict[str, tuple[int, ...]]]:
    texts: list[str] = []
    groups: dict[str, tuple[int, ...]] = {}
    for split, domains in ONLINE_CONTEXT_TEXTS.items():
        start = len(texts)
        for variants in domains:
            texts.extend(variants)
        groups[split] = tuple(range(start, len(texts)))
    return texts, groups
