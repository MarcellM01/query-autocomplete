from __future__ import annotations


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def dedupe_candidates(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    Drop exact duplicates after normalization. Prefix-style near-dedup is intentionally
    conservative here so mixed-length suggestions can coexist; tighten later if needed.
    """
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for text, sc in sorted(scored, key=lambda x: x[1], reverse=True):
        n = _norm(text)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append((text, sc))
    return out
