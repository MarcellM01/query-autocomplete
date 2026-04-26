from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


JSONValue: TypeAlias = None | bool | int | float | str | list[Any] | dict[str, Any]
JSONObject: TypeAlias = dict[str, JSONValue]


TOKEN_FLAG_NORMAL = 0
TOKEN_FLAG_PHRASE_HEAD = 1 << 0
TOKEN_FLAG_RARE = 1 << 1
TOKEN_FLAG_PROTECTED = 1 << 2

EDGE_TYPE_TOKEN = 0
EDGE_TYPE_PHRASE = 1


@dataclass(frozen=True)
class VocabEntry:
    offset: int
    length: int
    unigram_count: int
    doc_freq: int
    flags: int = TOKEN_FLAG_NORMAL


@dataclass(frozen=True)
class TokenPosting:
    token_id: int
    score_id: int


@dataclass(frozen=True)
class PhraseEntry:
    phrase_id: int
    token_ids: tuple[int, ...]
    display_length_chars: int
    flags: int = 0


@dataclass(frozen=True)
class ContextEdge:
    target_id: int
    target_type: int
    score_id: int


@dataclass(frozen=True)
class Document:
    text: str
    doc_id: str | None = None
    metadata: JSONObject = field(default_factory=dict)


@dataclass
class BinaryIndexData:
    vocab_strings: bytes
    vocab_entries: list[VocabEntry]
    serve_scores: list[int]
    token_postings: list[list[TokenPosting]]
    phrase_entries: list[PhraseEntry]
    context_edges: dict[tuple[int, ...], list[ContextEdge]]
    prefix_to_block: dict[str, int]
    scorer_payload: JSONObject = field(default_factory=dict)
    metadata: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreBreakdown:
    final_score: float
    prior_score: float
    scorer_score: float
    noise_penalty: float
    context_support_ratio: float
    context_support_penalty: float
    length_adjustment: float


@dataclass(frozen=True)
class PrefixMatchDiagnostic:
    fragment: str
    matched: str
    edit_distance: int
    fuzzy: bool


@dataclass(frozen=True)
class ExpansionStep:
    kind: str
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class SuggestionDiagnostic:
    text: str
    score: float
    breakdown: ScoreBreakdown
    diversity_group_key: str
    expansion_trace: tuple[ExpansionStep, ...]
    prefix_match: PrefixMatchDiagnostic | None = None
