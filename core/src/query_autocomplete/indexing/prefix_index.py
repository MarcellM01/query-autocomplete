from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol


class _RecordTrieLike(Protocol):
    def load(self, path: str) -> None: ...
    def save(self, path: str) -> None: ...
    def get(self, key: str, default: Any = ()) -> Any: ...
    def keys(self) -> Iterable[str]: ...


class PrefixIndex:
    """marisa-backed prefix -> postings block lookup."""

    def __init__(self) -> None:
        self._trie = self._new_trie({})
        self._alphabet: tuple[str, ...] = ()

    @staticmethod
    def _module():
        try:
            import marisa_trie
        except ImportError as exc:
            raise RuntimeError(
                "marisa-trie is required for PrefixIndex. Install the package dependency first."
            ) from exc
        return marisa_trie

    @classmethod
    def _new_trie(cls, mapping: dict[str, int]) -> _RecordTrieLike:
        marisa_trie = cls._module()
        items = [(prefix, (int(block_id),)) for prefix, block_id in sorted(mapping.items())]
        return marisa_trie.RecordTrie("I", items)

    def build(self, mapping: dict[str, int]) -> None:
        self._trie = self._new_trie(mapping)
        self._refresh_alphabet()

    def load(self, path: str | Path) -> None:
        trie = self._new_trie({})
        trie.load(str(path))
        self._trie = trie
        self._refresh_alphabet()

    def save(self, path: str | Path) -> None:
        self._trie.save(str(path))

    def get_block_id(self, prefix: str) -> int | None:
        if not prefix:
            return None
        values = self._trie.get(prefix, ())
        if not values:
            return None
        return int(values[0][0])

    def lookup_prefix_blocks(
        self,
        prefix: str,
        *,
        fuzzy_prefix: bool | str = "auto",
        max_edit_distance: int = 2,
    ) -> list[tuple[int, int, str]]:
        exact = self.get_block_id(prefix)
        if exact is not None:
            return [(exact, 0, prefix)]
        distance = self._effective_edit_distance(
            prefix,
            fuzzy_prefix=fuzzy_prefix,
            max_edit_distance=max_edit_distance,
        )
        if distance <= 0:
            return []
        out: dict[int, tuple[int, str]] = {}
        for candidate, candidate_distance in self._edit_neighbors(prefix, distance):
            block_id = self.get_block_id(candidate)
            if block_id is None:
                continue
            existing = out.get(block_id)
            if existing is None or (candidate_distance, candidate) < (existing[0], existing[1]):
                out[block_id] = (candidate_distance, candidate)
        return [
            (block_id, edit_distance, candidate)
            for block_id, (edit_distance, candidate) in sorted(
                out.items(),
                key=lambda item: (item[1][0], item[1][1], item[0]),
            )
        ]

    def to_dict(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key in self._trie.keys():
            out[key] = int(self._trie.get(key)[0][0])
        return out

    def _refresh_alphabet(self) -> None:
        chars = {char for key in self._trie.keys() for char in key}
        self._alphabet = tuple(sorted(chars))

    def _effective_edit_distance(
        self,
        prefix: str,
        *,
        fuzzy_prefix: bool | str,
        max_edit_distance: int,
    ) -> int:
        if not prefix or fuzzy_prefix is False or max_edit_distance <= 0:
            return 0
        if fuzzy_prefix == "auto" and len(prefix) < 3:
            return 0
        if fuzzy_prefix not in (True, "auto"):
            return 0
        capped = min(int(max_edit_distance), 2)
        if len(prefix) < 6:
            return min(capped, 1)
        return capped

    def _edit_neighbors(self, text: str, max_distance: int) -> Iterable[tuple[str, int]]:
        seen = {text}
        frontier = {text}
        for distance in range(1, max_distance + 1):
            next_frontier: set[str] = set()
            for item in frontier:
                for neighbor in self._one_edit_neighbors(item):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
                    yield neighbor, distance
            frontier = next_frontier

    def _one_edit_neighbors(self, text: str) -> Iterable[str]:
        length = len(text)
        for index in range(length):
            yield text[:index] + text[index + 1 :]
        for index in range(length - 1):
            if text[index] != text[index + 1]:
                yield text[:index] + text[index + 1] + text[index] + text[index + 2 :]
        for index in range(length):
            for char in self._alphabet:
                if char != text[index]:
                    yield text[:index] + char + text[index + 1 :]
        for index in range(length + 1):
            for char in self._alphabet:
                yield text[:index] + char + text[index:]
