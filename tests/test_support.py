from __future__ import annotations

import json
import types
from pathlib import Path


def fake_marisa_trie_module() -> types.SimpleNamespace:
    class RecordTrie:
        def __init__(self, _fmt: str, items=()) -> None:
            self._mapping = {key: tuple(values) for key, values in items}

        def keys(self, prefix: str = "") -> list[str]:
            return [key for key in sorted(self._mapping) if key.startswith(prefix)]

        def get(self, key: str, default=()):
            value = self._mapping.get(key)
            if value is None:
                return default
            return [value]

        def save(self, path: str) -> None:
            Path(path).write_text(json.dumps(self._mapping), encoding="utf-8")

        def load(self, path: str) -> None:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            self._mapping = {key: tuple(value) for key, value in raw.items()}

    return types.SimpleNamespace(RecordTrie=RecordTrie)
