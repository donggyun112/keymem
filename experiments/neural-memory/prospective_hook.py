from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_EVENT_PATH = Path(__file__).parent / "local-data" / "prospective-events.jsonl"
# Avoid importing neural_memory/__init__.py and PyTorch in a latency-sensitive hook.
sys.path.insert(0, str(Path(__file__).parent / "neural_memory"))
from prospective import ProspectiveLogger  # noqa: E402


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return 0
        configured = os.environ.get("NEURAL_MEMORY_EVENT_LOG")
        event_path = Path(configured).expanduser() if configured else DEFAULT_EVENT_PATH
        ProspectiveLogger(event_path).handle(payload)
    except Exception:
        # An observation failure must never alter or block the agent being observed.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
