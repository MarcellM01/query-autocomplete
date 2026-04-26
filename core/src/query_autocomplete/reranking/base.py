from __future__ import annotations


class BaseReranker:
    def rerank(self, prefix: str, candidates: list[str]) -> list[str]:
        return candidates
