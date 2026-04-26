from __future__ import annotations

import re

from query_autocomplete.reranking.base import BaseReranker

# 1–2 char fragment attached to a longer alphabetic token (e.g. "spain" + "k" → "spaink", bad edits).
_TYPO_SPOON_END = re.compile(r"([a-z]{2,20})([kK])$", re.IGNORECASE)
# Triple repeated letters (keyboard noise, rare in clean prose in continuations)
_TRIPLES = re.compile(r"(.)\1{2,}")


def _after_prefix(prefix: str, s: str) -> str:
    if not s:
        return ""
    if not prefix:
        return s
    p = len(prefix)
    if len(s) < p or s[:p].lower() != prefix[:p].lower():
        pr = prefix.rstrip()
        if s.lower().startswith(pr.lower()):
            return s[len(pr) :].lstrip()
        return s
    return s[p:].lstrip()


def _word_penalty(word: str) -> float:
    w = word.strip()
    if not w:
        return 0.0
    p = 0.0
    alnum = w.isalnum() and not w.isdigit()
    if alnum and len(w) == 1 and w.lower() not in ("a", "i"):
        p += 4.0
    if alnum and 5 <= len(w) <= 32:
        m = _TYPO_SPOON_END.search(w)
        if m and len(m.group(1)) >= 4:
            p += 1.2
    if alnum and _TRIPLES.search(w):
        p += 0.5
    return p


def _heuristic_key(prefix: str, candidate: str, original_index: int) -> tuple[float, int]:
    """Lower tuple sorts earlier (higher quality). original_index only breaks ties."""
    tail = _after_prefix(prefix, candidate)
    if not tail:
        return (0.0, original_index)
    p = 0.0
    for tok in tail.split():
        p += _word_penalty(tok)
    p += 0.015 * len(tail)
    p += 0.08 * len(tail.split())
    return (p, original_index)


class HeuristicReranker(BaseReranker):
    """
    Rerank suggested full strings with cheap surface heuristics on the *tail* (new text after
    the user prefix). Deprioritises very short free-standing tokens, likely typo/run-on
    characters at word ends, and long noisy tails, while keeping the model order as a tiebreak.
    """

    def rerank(self, prefix: str, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        items = list(enumerate(candidates))
        items.sort(key=lambda it: _heuristic_key(prefix, it[1], it[0]))
        return [c for _i, c in items]
