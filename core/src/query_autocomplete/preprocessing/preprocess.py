from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from importlib import import_module
from typing import Any

from query_autocomplete.config import NormalizationConfig
from query_autocomplete.models import Document

_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[._:/\\@+\-'][0-9a-z]+)*", re.IGNORECASE)

# Split after sentence-ending punctuation only when followed by an uppercase letter
# or digit — conservative enough to skip abbreviations like "e.g. something".
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@lru_cache(maxsize=16)
def _get_pysbd_segmenter(language: str) -> Any:
    pysbd = import_module("pysbd")
    return pysbd.Segmenter(language=language, clean=True)
_PUNCT_TRANSLATIONS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": " ",
        "\u00b7": ".",
        "\u2044": "/",
        "\u2215": "/",
        "\uff0f": "/",
        "\uff20": "@",
        "\uff0b": "+",
        "\uff3f": "_",
        "\uff0e": ".",
        "\u3002": ".",
    }
)


def _normalize_base(text: str, *, config: NormalizationConfig) -> str:
    value = text
    if config.unicode_nfkc:
        value = unicodedata.normalize("NFKC", value)
    value = value.translate(_PUNCT_TRANSLATIONS)
    if config.lowercase:
        value = value.lower()
    if config.strip_accents:
        value = "".join(
            ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)
        )
    return " ".join(value.split())


def tokenize_text(text: str, *, config: NormalizationConfig) -> list[str]:
    value = _normalize_base(text, config=config)
    if not value:
        return []
    if not config.strip_punctuation:
        return value.split()
    return [match.group(0) for match in _TOKEN_RE.finditer(value)]


def normalize_text(text: str, *, config: NormalizationConfig) -> str:
    tokens = tokenize_text(text, config=config)
    if tokens:
        return " ".join(tokens)
    return _normalize_base(text, config=config).strip()


def _iter_segments(text: str, *, split_sentences: bool, pysbd_language: str | None = None) -> list[str]:
    """Return a list of sub-line segments, optionally splitting on sentence boundaries."""
    segments: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if split_sentences:
            if pysbd_language is not None:
                segments.extend(_get_pysbd_segmenter(pysbd_language).segment(line))
            else:
                segments.extend(s.strip() for s in _SENTENCE_BOUNDARY_RE.split(line) if s.strip())
        else:
            segments.append(line)
    return segments


def preprocess_training_token_lines(
    documents: list[Document],
    *,
    config: NormalizationConfig,
) -> list[list[str]]:
    out: list[list[str]] = []
    for document in documents:
        for segment in _iter_segments(document.text, split_sentences=config.split_sentences, pysbd_language=config.pysbd_language):
            tokens = tokenize_text(segment, config=config)
            if tokens:
                out.append(tokens)
    return out


def preprocess_training_docs(
    documents: list[Document],
    *,
    config: NormalizationConfig,
) -> list[str]:
    return [" ".join(tokens) for tokens in preprocess_training_token_lines(documents, config=config)]
