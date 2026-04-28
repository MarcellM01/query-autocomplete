from __future__ import annotations

import threading
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from query_autocomplete.artifacts import (
    ARTIFACT_VERSION,
    reserve_default_artifact_directory,
    resolve_storage_directory,
)
from query_autocomplete.binary_format import (
    MANIFEST_FILENAME,
    PREFIX_FILENAME,
    load_binary_index,
    read_manifest,
)
from query_autocomplete.builder import compile_index, write_compiled_index
from query_autocomplete.config import BuildConfig, NormalizationConfig, QualityProfile, SuggestConfig, apply_quality_profile
from query_autocomplete.input_types import DocumentLike, coerce_documents
from query_autocomplete.indexing.prefix_index import PrefixIndex
from query_autocomplete.models import BinaryIndexData, Document
from query_autocomplete.reranking.base import BaseReranker
from query_autocomplete.runtime import RuntimeIndex
from query_autocomplete.preprocessing.preprocess import tokenize_text


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class Autocomplete:
    def __init__(
        self,
        documents: list[DocumentLike],
        *,
        build_config: BuildConfig | None = None,
        suggest_config: SuggestConfig | None = None,
        quality_profile: QualityProfile = "balanced",
    ) -> None:
        self._documents = coerce_documents(documents)
        self._build_config, self._suggest_config = apply_quality_profile(
            quality_profile,
            build_config=build_config,
            suggest_config=suggest_config,
        )
        self._runtime: RuntimeIndex | None = None
        self._compiled: BinaryIndexData | None = None
        self._prefix_index = PrefixIndex()
        self._manifest: dict[str, Any] | None = None
        self._built = False
        self._promotable_documents = True
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        documents: list[DocumentLike],
        *,
        build_config: BuildConfig | None = None,
        suggest_config: SuggestConfig | None = None,
        quality_profile: QualityProfile = "balanced",
        max_generated_words: int | None = None,
        phrase_min_count: int | None = None,
    ) -> Autocomplete:
        engine = cls(
            documents,
            build_config=build_config,
            suggest_config=suggest_config,
            quality_profile=quality_profile,
        )
        engine._build(max_generated_words=max_generated_words, phrase_min_count=phrase_min_count)
        return engine

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> Autocomplete:
        artifact_dir = resolve_storage_directory(path)
        if not artifact_dir.exists() or not artifact_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
        manifest = read_manifest(artifact_dir / MANIFEST_FILENAME)
        config_data = dict(manifest["config"])
        build_data = dict(config_data.get("build", {}))
        suggest_data = dict(config_data.get("suggest", {}))
        normalization_data = build_data.get("normalization", {})
        build_config = BuildConfig(**{**build_data, "normalization": NormalizationConfig(**normalization_data)})
        suggest_config = SuggestConfig(**suggest_data)
        data = load_binary_index(artifact_dir, prefix_to_block={})
        engine = cls._from_compiled(
            data,
            build_config=build_config,
            suggest_config=suggest_config,
        )
        engine._prefix_index.load(artifact_dir / manifest["files"].get("prefix_trie", PREFIX_FILENAME))
        if engine._runtime is not None:
            engine._runtime = RuntimeIndex(
                data,
                build_config=build_config,
                suggest_config=suggest_config,
                prefix_index=engine._prefix_index,
            )
        engine._manifest = manifest
        engine._promotable_documents = False
        return engine

    def _build(
        self,
        max_generated_words: int | None = None,
        phrase_min_count: int | None = None,
        *,
        build_config: BuildConfig | None = None,
    ) -> None:
        with self._lock:
            if not self._documents:
                raise ValueError("Cannot build with no documents; provide at least one non-empty source.")
            base_config = build_config if build_config is not None else self._build_config
            overrides = {
                "max_generated_words": max(
                    1,
                    int(max_generated_words if max_generated_words is not None else base_config.max_generated_words),
                )
            }
            if phrase_min_count is not None:
                overrides["phrase_min_count"] = max(1, int(phrase_min_count))
            config = replace(base_config, **overrides)
            compiled, prefix_index, _stats = compile_index(self._documents, build_config=config, suggest_config=self._suggest_config)
            compiled.metadata["config"] = {"build": asdict(config), "suggest": asdict(self._suggest_config)}
            self._build_config = config
            self._compiled = compiled
            self._runtime = RuntimeIndex(
                compiled,
                build_config=config,
                suggest_config=self._suggest_config,
                prefix_index=prefix_index,
            )
            self._prefix_index = prefix_index
            self._manifest = compiled.metadata
            self._built = True

    def export_documents(self) -> list[Document]:
        return [replace(document) for document in self._documents]

    def save(self, path: str | Path | None = None) -> None:
        with self._lock:
            if not self._built or self._compiled is None:
                raise RuntimeError("Call build() before save().")
            output_dir = reserve_default_artifact_directory(self._documents) if path is None else resolve_storage_directory(path)
            write_compiled_index(self._compiled, prefix_index=self._prefix_index, output_dir=output_dir)

    def warm(self, sample_query: str = "a", *, topk: int = 1) -> None:
        """Initialize runtime query structures before serving real traffic."""
        self.suggest(sample_query, topk=topk)

    def suggest(
        self,
        text: str,
        *,
        topk: int | None = None,
        max_words: int | None = None,
        length_bias: float | None = None,
        collapse_prefix_ladders: bool | None = None,
        suggest_config: SuggestConfig | None = None,
        reranker: BaseReranker | None = None,
    ) -> list[str]:
        with self._lock:
            if not self._built or self._runtime is None:
                raise RuntimeError("Call build() before suggest().")
            effective_suggest_config = suggest_config if suggest_config is not None else self._suggest_config
            if collapse_prefix_ladders is not None:
                effective_suggest_config = replace(
                    effective_suggest_config,
                    collapse_prefix_ladders=bool(collapse_prefix_ladders),
                )
            effective_reranker = reranker or BaseReranker()
            tk = max(1, int(topk if topk is not None else effective_suggest_config.default_top_k))
            mw = max(1, int(max_words if max_words is not None else effective_suggest_config.max_suggestion_words))
            self._warn_if_suggest_exceeds_build_limits(
                text,
                max_words=mw,
                suggest_config=effective_suggest_config,
            )
            bias = _clamp01(float(length_bias if length_bias is not None else effective_suggest_config.default_length_bias))
            ordered = self._runtime.suggest(
                text,
                topk=tk,
                max_words=mw,
                prefer_long=bias,
                suggest_config=effective_suggest_config,
            )
            return effective_reranker.rerank(text, ordered)[:tk]

    def inspect(
        self,
        text: str,
        *,
        topk: int | None = None,
        max_words: int | None = None,
        length_bias: float | None = None,
        collapse_prefix_ladders: bool | None = None,
        suggest_config: SuggestConfig | None = None,
        reranker: BaseReranker | None = None,
    ):
        with self._lock:
            if not self._built or self._runtime is None:
                raise RuntimeError("Call build() before inspect().")
            effective_suggest_config = suggest_config if suggest_config is not None else self._suggest_config
            if collapse_prefix_ladders is not None:
                effective_suggest_config = replace(
                    effective_suggest_config,
                    collapse_prefix_ladders=bool(collapse_prefix_ladders),
                )
            effective_reranker = reranker or BaseReranker()
            tk = max(1, int(topk if topk is not None else effective_suggest_config.default_top_k))
            mw = max(1, int(max_words if max_words is not None else effective_suggest_config.max_suggestion_words))
            self._warn_if_suggest_exceeds_build_limits(
                text,
                max_words=mw,
                suggest_config=effective_suggest_config,
            )
            bias = _clamp01(float(length_bias if length_bias is not None else effective_suggest_config.default_length_bias))
            diagnostics = self._runtime.inspect(
                text,
                topk=tk,
                max_words=mw,
                prefer_long=bias,
                suggest_config=effective_suggest_config,
            )
            ordered_texts = effective_reranker.rerank(text, [item.text for item in diagnostics])[:tk]
            by_text = {item.text: item for item in diagnostics}
            return [by_text[item] for item in ordered_texts if item in by_text]

    @classmethod
    def _from_compiled(
        cls,
        compiled: BinaryIndexData,
        *,
        build_config: BuildConfig,
        suggest_config: SuggestConfig,
    ) -> Autocomplete:
        engine = cls([], build_config=build_config, suggest_config=suggest_config)
        prefix_index = PrefixIndex()
        prefix_index.build(compiled.prefix_to_block)
        engine._runtime = RuntimeIndex(
            compiled,
            build_config=build_config,
            suggest_config=suggest_config,
            prefix_index=prefix_index,
        )
        engine._compiled = compiled
        engine._prefix_index = prefix_index
        engine._manifest = compiled.metadata
        engine._built = True
        return engine

    def _warn_if_suggest_exceeds_build_limits(
        self,
        text: str,
        *,
        max_words: int,
        suggest_config: SuggestConfig,
    ) -> None:
        if max_words > self._build_config.max_generated_words:
            warnings.warn(
                "suggest(max_words=...) exceeds the index build budget: "
                f"max_words={max_words}, BuildConfig.max_generated_words={self._build_config.max_generated_words}. "
                "Rebuild with a larger max_generated_words value if you want longer continuations to be part of the "
                "build-time artifact budget.",
                UserWarning,
                stacklevel=3,
            )
        fragment = self._query_fragment(text)
        if fragment and len(fragment) > self._build_config.max_indexed_prefix_chars:
            warnings.warn(
                "The current query fragment is longer than the indexed prefix budget: "
                f"fragment length={len(fragment)}, BuildConfig.max_indexed_prefix_chars="
                f"{self._build_config.max_indexed_prefix_chars}. Partial-token lookup may not find completions "
                "past that prefix length; rebuild with a larger max_indexed_prefix_chars value for deeper "
                "partial-token matching.",
                UserWarning,
                stacklevel=3,
            )
        if suggest_config.token_branch_limit > self._build_config.top_next_tokens:
            warnings.warn(
                "SuggestConfig.token_branch_limit exceeds the token fanout stored at build time: "
                f"token_branch_limit={suggest_config.token_branch_limit}, BuildConfig.top_next_tokens="
                f"{self._build_config.top_next_tokens}. Rebuild with a larger top_next_tokens value if you want "
                "more token branches available at serving time.",
                UserWarning,
                stacklevel=3,
            )
        if suggest_config.phrase_branch_limit > self._build_config.top_next_phrases:
            warnings.warn(
                "SuggestConfig.phrase_branch_limit exceeds the phrase fanout stored at build time: "
                f"phrase_branch_limit={suggest_config.phrase_branch_limit}, BuildConfig.top_next_phrases="
                f"{self._build_config.top_next_phrases}. Rebuild with a larger top_next_phrases value if you want "
                "more phrase branches available at serving time.",
                UserWarning,
                stacklevel=3,
            )

    def _query_fragment(self, text: str) -> str:
        if not text or text[-1].isspace():
            return ""
        tokens = tokenize_text(text, config=self._build_config.normalization)
        return tokens[-1] if tokens else ""


__all__ = ["Autocomplete", "ARTIFACT_VERSION", "_clamp01"]
