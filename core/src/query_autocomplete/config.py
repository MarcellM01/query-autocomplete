from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from importlib import import_module
from typing import Any, Literal


QualityProfile = Literal["balanced", "precision", "recall", "code_or_logs", "natural_language"]
_UNSET = object()


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = True
    unicode_nfkc: bool = True
    strip_accents: bool = False
    strip_punctuation: bool = True
    split_sentences: bool = True
    # When set to a pysbd language code (e.g. "en", "de", "fr"), pysbd is used for
    # sentence splitting instead of the built-in regex. Only active when split_sentences=True.
    pysbd_language: str | None = None

    def __post_init__(self) -> None:
        if self.pysbd_language is None:
            return
        if not self.split_sentences:
            return
        try:
            pysbd = import_module("pysbd")
            pysbd.Segmenter(language=self.pysbd_language, clean=True)
        except ImportError as exc:
            raise ImportError(
                "NormalizationConfig.pysbd_language requires the optional 'pysbd' dependency. "
                "Install query-autocomplete[chunking] or query-autocomplete[all]."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - pysbd raises package-specific exceptions.
            raise ValueError(f"Unsupported pysbd language code: {self.pysbd_language!r}") from exc


@dataclass(frozen=True)
class BuildConfig:
    max_generated_words: int = 4
    max_indexed_prefix_chars: int = 24
    max_context_tokens: int = 3
    top_tokens_per_prefix: int = 64
    top_next_tokens: int = 32
    top_next_phrases: int = 16
    phrase_min_count: int = 2
    phrase_min_doc_freq: int = 1
    phrase_min_pmi: float = 0.0
    phrase_max_dominant_extension_ratio: float = 0.95
    phrase_boundary_generic_min_count: int = 8
    phrase_max_len: int = 4
    # When None, vocabulary trimming is disabled. When set, we apply min-count filtering only
    # if the corpus has at least this many total token positions (sum of line lengths) so
    # small corpora keep rare types; large ones drop the long tail.
    vocab_prune_min_total_tokens: int | None = 100_000
    # Types with unigram count strictly below this are dropped from training material when
    # pruning is active (e.g. 2 removes singletons / hapax legomena).
    vocab_prune_min_unigram_count: int = 2
    # Minimum number of *training lines* (sentence/line segments) a type must appear in.
    # When >1, types that occur only in one segment are dropped even if unigram count >=
    # vocab_prune_min_unigram_count (e.g. repeated inside one sentence) — reduces spammy locals.
    # Only enforced when the corpus has at least vocab_prune_line_count_to_apply_df lines (see next).
    vocab_prune_min_segment_freq: int = 2
    # If unigram count is >= this, the type is always kept, even if segment_freq is too low
    # (e.g. a very frequent word in one long line). Set to 0 to disable.
    vocab_prune_rescue_unigram: int = 12
    # If the line count is below this, segment-frequency rules are skipped (unigram rules only).
    # Protects tiny corpora and unit tests. Set to 0 to always apply segment rules when they apply.
    vocab_prune_line_count_to_apply_df: int = 5_000
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    def __init__(
        self,
        max_generated_words: int | object = _UNSET,
        max_indexed_prefix_chars: int | object = _UNSET,
        max_context_tokens: int | object = _UNSET,
        top_tokens_per_prefix: int | object = _UNSET,
        top_next_tokens: int | object = _UNSET,
        top_next_phrases: int | object = _UNSET,
        phrase_min_count: int | object = _UNSET,
        phrase_min_doc_freq: int | object = _UNSET,
        phrase_min_pmi: float | object = _UNSET,
        phrase_max_dominant_extension_ratio: float | object = _UNSET,
        phrase_boundary_generic_min_count: int | object = _UNSET,
        phrase_max_len: int | object = _UNSET,
        vocab_prune_min_total_tokens: int | None | object = _UNSET,
        vocab_prune_min_unigram_count: int | object = _UNSET,
        vocab_prune_min_segment_freq: int | object = _UNSET,
        vocab_prune_rescue_unigram: int | object = _UNSET,
        vocab_prune_line_count_to_apply_df: int | object = _UNSET,
        normalization: NormalizationConfig | object = _UNSET,
    ) -> None:
        raw_values = {
            "max_generated_words": max_generated_words,
            "max_indexed_prefix_chars": max_indexed_prefix_chars,
            "max_context_tokens": max_context_tokens,
            "top_tokens_per_prefix": top_tokens_per_prefix,
            "top_next_tokens": top_next_tokens,
            "top_next_phrases": top_next_phrases,
            "phrase_min_count": phrase_min_count,
            "phrase_min_doc_freq": phrase_min_doc_freq,
            "phrase_min_pmi": phrase_min_pmi,
            "phrase_max_dominant_extension_ratio": phrase_max_dominant_extension_ratio,
            "phrase_boundary_generic_min_count": phrase_boundary_generic_min_count,
            "phrase_max_len": phrase_max_len,
            "vocab_prune_min_total_tokens": vocab_prune_min_total_tokens,
            "vocab_prune_min_unigram_count": vocab_prune_min_unigram_count,
            "vocab_prune_min_segment_freq": vocab_prune_min_segment_freq,
            "vocab_prune_rescue_unigram": vocab_prune_rescue_unigram,
            "vocab_prune_line_count_to_apply_df": vocab_prune_line_count_to_apply_df,
            "normalization": normalization,
        }
        defaults: dict[str, Any] = {
            "max_generated_words": 4,
            "max_indexed_prefix_chars": 24,
            "max_context_tokens": 3,
            "top_tokens_per_prefix": 64,
            "top_next_tokens": 32,
            "top_next_phrases": 16,
            "phrase_min_count": 2,
            "phrase_min_doc_freq": 1,
            "phrase_min_pmi": 0.0,
            "phrase_max_dominant_extension_ratio": 0.95,
            "phrase_boundary_generic_min_count": 8,
            "phrase_max_len": 4,
            "vocab_prune_min_total_tokens": 100_000,
            "vocab_prune_min_unigram_count": 2,
            "vocab_prune_min_segment_freq": 2,
            "vocab_prune_rescue_unigram": 12,
            "vocab_prune_line_count_to_apply_df": 5_000,
            "normalization": NormalizationConfig(),
        }
        explicit = frozenset(name for name, value in raw_values.items() if value is not _UNSET)
        values = {
            name: (defaults[name] if value is _UNSET else value)
            for name, value in raw_values.items()
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_explicit_fields", explicit)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.max_context_tokens > 6:
            raise ValueError(
                "BuildConfig.max_context_tokens must be at most 6 because the current binary format stores "
                "context keys with a maximum width of 6."
            )


@dataclass(frozen=True)
class SuggestConfig:
    default_top_k: int = 10
    default_length_bias: float = 0.5
    max_suggestion_words: int = 4
    beam_width: int = 24
    token_branch_limit: int = 8
    phrase_branch_limit: int = 8
    prior_weight: float = 0.35
    noise_penalty_weight: float = 0.35
    suppress_redundant_continuations: bool = True
    min_context_support_ratio: float = 0.0
    context_support_penalty_weight: float = 0.25
    collapse_prefix_ladders: bool = True
    collapse_prefix_ladder_strategy: Literal["best", "prefer_longest", "prefer_shortest"] = "best"
    unknown_context_strategy: Literal["strict", "skip"] = "skip"
    normalize_phrase_scores_by_length: bool = False
    fuzzy_prefix: bool | Literal["auto"] = "auto"
    max_edit_distance: int = 2

    def __init__(
        self,
        default_top_k: int | object = _UNSET,
        default_length_bias: float | object = _UNSET,
        max_suggestion_words: int | object = _UNSET,
        beam_width: int | object = _UNSET,
        token_branch_limit: int | object = _UNSET,
        phrase_branch_limit: int | object = _UNSET,
        prior_weight: float | object = _UNSET,
        noise_penalty_weight: float | object = _UNSET,
        suppress_redundant_continuations: bool | object = _UNSET,
        min_context_support_ratio: float | object = _UNSET,
        context_support_penalty_weight: float | object = _UNSET,
        collapse_prefix_ladders: bool | object = _UNSET,
        collapse_prefix_ladder_strategy: Literal["best", "prefer_longest", "prefer_shortest"] | object = _UNSET,
        unknown_context_strategy: Literal["strict", "skip"] | object = _UNSET,
        normalize_phrase_scores_by_length: bool | object = _UNSET,
        fuzzy_prefix: bool | Literal["auto"] | object = _UNSET,
        max_edit_distance: int | object = _UNSET,
    ) -> None:
        raw_values = {
            "default_top_k": default_top_k,
            "default_length_bias": default_length_bias,
            "max_suggestion_words": max_suggestion_words,
            "beam_width": beam_width,
            "token_branch_limit": token_branch_limit,
            "phrase_branch_limit": phrase_branch_limit,
            "prior_weight": prior_weight,
            "noise_penalty_weight": noise_penalty_weight,
            "suppress_redundant_continuations": suppress_redundant_continuations,
            "min_context_support_ratio": min_context_support_ratio,
            "context_support_penalty_weight": context_support_penalty_weight,
            "collapse_prefix_ladders": collapse_prefix_ladders,
            "collapse_prefix_ladder_strategy": collapse_prefix_ladder_strategy,
            "unknown_context_strategy": unknown_context_strategy,
            "normalize_phrase_scores_by_length": normalize_phrase_scores_by_length,
            "fuzzy_prefix": fuzzy_prefix,
            "max_edit_distance": max_edit_distance,
        }
        defaults: dict[str, Any] = {
            "default_top_k": 10,
            "default_length_bias": 0.5,
            "max_suggestion_words": 4,
            "beam_width": 24,
            "token_branch_limit": 8,
            "phrase_branch_limit": 8,
            "prior_weight": 0.35,
            "noise_penalty_weight": 0.35,
            "suppress_redundant_continuations": True,
            "min_context_support_ratio": 0.0,
            "context_support_penalty_weight": 0.25,
            "collapse_prefix_ladders": True,
            "collapse_prefix_ladder_strategy": "best",
            "unknown_context_strategy": "skip",
            "normalize_phrase_scores_by_length": False,
            "fuzzy_prefix": "auto",
            "max_edit_distance": 2,
        }
        explicit = frozenset(name for name, value in raw_values.items() if value is not _UNSET)
        values = {
            name: (defaults[name] if value is _UNSET else value)
            for name, value in raw_values.items()
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_explicit_fields", explicit)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.fuzzy_prefix not in (True, False, "auto"):
            raise ValueError("SuggestConfig.fuzzy_prefix must be True, False, or 'auto'.")
        if int(self.max_edit_distance) < 0:
            raise ValueError("SuggestConfig.max_edit_distance must be non-negative.")


_PROFILE_DEFAULTS: dict[QualityProfile, tuple[dict[str, object], dict[str, object]]] = {
    "balanced": (
        {},
        {},
    ),
    "precision": (
        {
            "phrase_min_count": 3,
            "phrase_min_doc_freq": 2,
            "phrase_min_pmi": 0.15,
            "phrase_max_dominant_extension_ratio": 0.8,
            "vocab_prune_min_unigram_count": 3,
        },
        {
            "beam_width": 18,
            "token_branch_limit": 6,
            "phrase_branch_limit": 6,
            "noise_penalty_weight": 0.6,
            "context_support_penalty_weight": 0.45,
            "min_context_support_ratio": 0.25,
            "collapse_prefix_ladders": True,
        },
    ),
    "recall": (
        {
            "phrase_min_count": 2,
            "phrase_min_doc_freq": 1,
            "phrase_min_pmi": -0.25,
            "phrase_max_dominant_extension_ratio": 1.0,
            "vocab_prune_min_total_tokens": None,
        },
        {
            "beam_width": 32,
            "token_branch_limit": 12,
            "phrase_branch_limit": 12,
            "noise_penalty_weight": 0.15,
            "context_support_penalty_weight": 0.1,
            "collapse_prefix_ladders": False,
        },
    ),
    "code_or_logs": (
        {
            "phrase_min_count": 2,
            "phrase_min_doc_freq": 1,
            "phrase_min_pmi": -0.1,
            "phrase_boundary_generic_min_count": 4,
            "max_indexed_prefix_chars": 32,
        },
        {
            "beam_width": 28,
            "token_branch_limit": 10,
            "phrase_branch_limit": 8,
            "noise_penalty_weight": 0.05,
            "context_support_penalty_weight": 0.15,
            "collapse_prefix_ladders": False,
        },
    ),
    "natural_language": (
        {
            "phrase_min_count": 3,
            "phrase_min_doc_freq": 2,
            "phrase_min_pmi": 0.25,
            "phrase_max_dominant_extension_ratio": 0.82,
        },
        {
            "beam_width": 24,
            "token_branch_limit": 8,
            "phrase_branch_limit": 10,
            "noise_penalty_weight": 0.5,
            "context_support_penalty_weight": 0.35,
            "collapse_prefix_ladders": True,
        },
    ),
}


def apply_quality_profile(
    quality_profile: QualityProfile = "balanced",
    *,
    build_config: BuildConfig | None = None,
    suggest_config: SuggestConfig | None = None,
) -> tuple[BuildConfig, SuggestConfig]:
    try:
        build_defaults, suggest_defaults = _PROFILE_DEFAULTS[quality_profile]
    except KeyError as exc:
        raise ValueError(f"Unknown quality profile: {quality_profile!r}") from exc
    build = (
        _merge_profile_config(BuildConfig(), build_defaults, build_config)
        if build_config is not None
        else replace(BuildConfig(), **build_defaults)
    )
    suggest = (
        _merge_profile_config(SuggestConfig(), suggest_defaults, suggest_config)
        if suggest_config is not None
        else replace(SuggestConfig(), **suggest_defaults)
    )
    return build, suggest


def _merge_profile_config(default_config, profile_defaults: dict[str, object], explicit_config):
    profiled = replace(default_config, **profile_defaults)
    explicit_fields = getattr(explicit_config, "_explicit_fields", None)
    if explicit_fields is not None:
        overrides = {
            item.name: getattr(explicit_config, item.name)
            for item in fields(default_config)
            if item.name in explicit_fields
        }
        return replace(profiled, **overrides)
    overrides = {
        item.name: getattr(explicit_config, item.name)
        for item in fields(default_config)
        if getattr(explicit_config, item.name) != getattr(default_config, item.name)
    }
    return replace(profiled, **overrides)
