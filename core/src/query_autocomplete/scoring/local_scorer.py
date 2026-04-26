from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


class LocalNgramScorer:
    """Interpolated Kneser-Ney bigram scorer built from local token counts."""

    def __init__(
        self,
        *,
        unigrams: Counter[str] | None = None,
        bigrams: Counter[tuple[str, str]] | None = None,
        discount: float = 0.75,
    ) -> None:
        self._discount = float(discount)
        self._u = Counter(unigrams or {})
        self._b = Counter(bigrams or {})
        self._refresh()

    @classmethod
    def from_lines(cls, lines: list[list[str]], *, discount: float = 0.75) -> LocalNgramScorer:
        unigrams: Counter[str] = Counter()
        bigrams: Counter[tuple[str, str]] = Counter()
        for tokens in lines:
            if not tokens:
                continue
            unigrams.update(tokens)
            for left, right in zip(tokens, tokens[1:]):
                bigrams[(left, right)] += 1
        return cls(unigrams=unigrams, bigrams=bigrams, discount=discount)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LocalNgramScorer:
        if not isinstance(payload, dict):
            raise ValueError("Invalid scorer artifact: expected an object.")

        if payload.get("type") != "interpolated_kneser_ney" or int(payload.get("version", 0)) != 1:
            raise ValueError("Invalid scorer artifact: unsupported scorer type or version.")
        unigram_data = payload.get("unigrams")
        bigram_data = payload.get("bigrams")
        discount = float(payload.get("discount", 0.75))
        if not isinstance(unigram_data, dict) or not isinstance(bigram_data, dict):
            raise ValueError("Invalid scorer artifact: missing n-gram counts.")

        bigrams: Counter[tuple[str, str]] = Counter()
        for key, value in bigram_data.items():
            if not isinstance(key, str) or "\t" not in key:
                raise ValueError("Invalid scorer artifact: malformed bigram key.")
            left, right = key.split("\t", 1)
            bigrams[(left, right)] = int(value)

        return cls(
            unigrams=Counter({token: int(count) for token, count in unigram_data.items()}),
            bigrams=bigrams,
            discount=discount,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "interpolated_kneser_ney",
            "version": 1,
            "discount": self._discount,
            "unigrams": dict(sorted(self._u.items())),
            "bigrams": {
                f"{left}\t{right}": int(count)
                for (left, right), count in sorted(self._b.items())
            },
        }

    def merge(self, other: LocalNgramScorer) -> LocalNgramScorer:
        merged = LocalNgramScorer(discount=self._discount)
        merged._u = self._u + other._u
        merged._b = self._b + other._b
        merged._refresh()
        return merged

    def score(self, text: str) -> float:
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return 0.0

        total = 0.0
        for index, token in enumerate(tokens):
            if index == 0:
                probability = self._unigram_probability(token)
            else:
                prev = tokens[index - 1]
                probability = self._bigram_probability(prev, token)
            total += math.log(probability + 1e-12)
        return total

    @property
    def unigrams(self) -> Counter[str]:
        return Counter(self._u)

    @property
    def bigrams(self) -> Counter[tuple[str, str]]:
        return Counter(self._b)

    def _refresh(self) -> None:
        self._total_unigrams = sum(self._u.values()) or 1
        self._vocab_size = max(len(self._u), 1)
        followers: dict[str, set[str]] = defaultdict(set)
        predecessors: dict[str, set[str]] = defaultdict(set)
        for left, right in self._b:
            followers[left].add(right)
            predecessors[right].add(left)
        self._unique_followers = {token: len(values) for token, values in followers.items()}
        self._continuation_counts = {token: len(values) for token, values in predecessors.items()}
        self._unique_bigram_count = max(len(self._b), 1)

    def _unigram_probability(self, token: str) -> float:
        count = self._u.get(token, 0)
        if count <= 0:
            return 1.0 / (self._total_unigrams + self._vocab_size)
        return count / self._total_unigrams

    def _continuation_probability(self, token: str) -> float:
        count = self._continuation_counts.get(token, 0)
        if count <= 0:
            return 1.0 / (self._unique_bigram_count + self._vocab_size)
        return count / self._unique_bigram_count

    def _bigram_probability(self, prev: str, token: str) -> float:
        prev_count = self._u.get(prev, 0)
        if prev_count <= 0:
            return self._continuation_probability(token)
        bigram_count = self._b.get((prev, token), 0)
        discounted = max(bigram_count - self._discount, 0.0) / prev_count
        lambda_weight = (self._discount * self._unique_followers.get(prev, 0)) / prev_count
        return discounted + (lambda_weight * self._continuation_probability(token))
