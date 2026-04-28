from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace as dataclass_replace
from sys import getsizeof

from query_autocomplete.builder import compile_index
from query_autocomplete.config import BuildConfig, QualityProfile, SuggestConfig, apply_quality_profile
from query_autocomplete.adaptive_storage import BaseStore, CompiledIndexRecord, DuplicateDocument, StoreRecord, open_store
from query_autocomplete.engine import Autocomplete
from query_autocomplete.input_types import DocumentLike, coerce_documents
from query_autocomplete.models import Document
from query_autocomplete.reranking.base import BaseReranker


_ENGINE_CACHE_LOCK = threading.Lock()
_ENGINE_CACHE_MAX_SIZE = 8
_ENGINE_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _compiled_index_approx_bytes(engine: Autocomplete) -> int:
    compiled = getattr(engine, "_compiled", None)
    if compiled is None:
        return 0
    total = getsizeof(compiled)
    total += len(compiled.vocab_strings)
    total += sum(getsizeof(entry) for entry in compiled.vocab_entries)
    total += getsizeof(compiled.serve_scores) + (len(compiled.serve_scores) * 4)
    total += getsizeof(compiled.token_postings)
    total += sum(getsizeof(block) + (len(block) * 16) for block in compiled.token_postings)
    total += getsizeof(compiled.phrase_entries) + sum(getsizeof(phrase) + (len(phrase.token_ids) * 4) for phrase in compiled.phrase_entries)
    total += getsizeof(compiled.context_edges)
    total += sum(getsizeof(key) + getsizeof(edges) + (len(edges) * 16) for key, edges in compiled.context_edges.items())
    total += getsizeof(compiled.prefix_to_block) + sum(len(key.encode("utf-8")) + 8 for key in compiled.prefix_to_block)
    total += getsizeof(compiled.scorer_payload) + getsizeof(compiled.metadata)
    return max(1, total)


class _EngineCache:
    def __init__(self, *, max_size: int, max_bytes: int) -> None:
        self._entries: OrderedDict[tuple[str, int], tuple[Autocomplete, int]] = OrderedDict()
        self._max_size = max(1, int(max_size))
        self._max_bytes = max(1, int(max_bytes))
        self._bytes = 0

    def get(self, key: tuple[str, int]) -> Autocomplete | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        engine, size = entry
        self._entries.move_to_end(key)
        return engine

    def put(self, key: tuple[str, int], engine: Autocomplete) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= old[1]
        size = _compiled_index_approx_bytes(engine)
        self._entries[key] = (engine, size)
        self._bytes += size
        self._evict()

    def invalidate(self, predicate) -> None:
        stale_keys = [key for key in self._entries if predicate(key)]
        for key in stale_keys:
            _engine, size = self._entries.pop(key)
            self._bytes -= size

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def byte_size(self) -> int:
        return self._bytes

    def _evict(self) -> None:
        while self._entries and (len(self._entries) > self._max_size or self._bytes > self._max_bytes):
            _key, (_engine, size) = self._entries.popitem(last=False)
            self._bytes -= size


_ENGINE_CACHE = _EngineCache(max_size=_ENGINE_CACHE_MAX_SIZE, max_bytes=_ENGINE_CACHE_MAX_BYTES)


@dataclass(frozen=True)
class IngestResult:
    inserted_count: int
    duplicate_doc_ids: tuple[str, ...]
    duplicate_texts: tuple[DuplicateDocument, ...]


@dataclass(frozen=True)
class DeleteResult:
    deleted_count: int


class AdaptiveStore:
    def __init__(
        self,
        store: BaseStore,
        *,
        record: StoreRecord | None = None,
    ) -> None:
        self._store = store
        self._record = record
        self._current_index_record: CompiledIndexRecord | None = None
        self._lock = threading.RLock()

    @classmethod
    def open(
        cls,
        store_url: str,
        *,
        build_config: BuildConfig | None = None,
        suggest_config: SuggestConfig | None = None,
        quality_profile: QualityProfile = "balanced",
        phrase_min_count: int | None = None,
    ) -> AdaptiveStore:
        store = open_store(store_url)
        store.create_schema()
        build, suggest = apply_quality_profile(
            quality_profile,
            build_config=build_config,
            suggest_config=suggest_config,
        )
        if phrase_min_count is not None:
            build = dataclass_replace(build, phrase_min_count=max(1, int(phrase_min_count)))
        record = store.open_record(build_config=build, suggest_config=suggest)
        if (build_config is not None or suggest_config is not None or phrase_min_count is not None) and (
            record.build_config != build or record.suggest_config != suggest
        ):
            raise ValueError("Adaptive store already exists with a different config.")
        return cls(store, record=record)

    @classmethod
    def open_or_create(
        cls,
        store_url: str,
        *,
        build_config: BuildConfig | None = None,
        suggest_config: SuggestConfig | None = None,
        quality_profile: QualityProfile = "balanced",
        phrase_min_count: int | None = None,
    ) -> AdaptiveStore:
        return cls.open(
            store_url,
            build_config=build_config,
            suggest_config=suggest_config,
            quality_profile=quality_profile,
            phrase_min_count=phrase_min_count,
        )

    @classmethod
    def import_autocomplete(
        cls,
        store_url: str,
        *,
        engine: Autocomplete,
    ) -> AdaptiveStore:
        documents = engine.export_documents()
        if not documents or not getattr(engine, "_promotable_documents", False):
            raise ValueError(
                "Cannot promote this Autocomplete instance to SQL because no source documents are attached. "
                "Build from source docs or export documents from an in-memory Autocomplete."
            )
        store = cls.open(
            store_url,
            build_config=engine._build_config,
            suggest_config=engine._suggest_config,
        )
        store.add_documents(documents)
        return store

    def add_documents(self, documents: list[DocumentLike]) -> IngestResult:
        with self._lock:
            eligible = [self._prepare_document(doc) for doc in coerce_documents(documents) if doc.text]
            if not eligible:
                return IngestResult(inserted_count=0, duplicate_doc_ids=(), duplicate_texts=())
            self._ensure_record()
            t0 = time.perf_counter()
            result = self._store.insert_documents(eligible)
            ingest_ms = int((time.perf_counter() - t0) * 1000)
            self._store.insert_ingest_log(attempted=len(eligible), inserted=result.inserted_count, ingest_ms=ingest_ms)
            self._record = self._store.load_record()
            if result.inserted_count > 0:
                self._mark_serving_artifacts_stale()
            return IngestResult(
                inserted_count=result.inserted_count,
                duplicate_doc_ids=result.duplicate_doc_ids,
                duplicate_texts=result.duplicate_texts,
            )

    def remove_document(self, doc_id: str) -> DeleteResult:
        with self._lock:
            if self._record is None:
                return DeleteResult(deleted_count=0)
            result = self._store.delete_document(doc_id)
            if result.deleted_count == 0:
                return DeleteResult(deleted_count=0)
            if self._store.fetch_all_documents().count == 0:
                self._store.clear_current_index()
                self._record = self._store.load_record()
                self._current_index_record = None
                self._invalidate_cached_engines()
                return DeleteResult(deleted_count=result.deleted_count)
            self._record = self._store.load_record()
            self._current_index_record = None
            self._mark_serving_artifacts_stale()
            return DeleteResult(deleted_count=result.deleted_count)

    def list_documents(self) -> list[Document]:
        with self._lock:
            if self._record is None:
                return []
            return self._store.fetch_all_documents().documents

    def clear(self) -> None:
        with self._lock:
            self._ensure_record()
            self._store.reset(build_config=self._build_config(), suggest_config=self._suggest_config())
            self._record = self._store.load_record()
            self._current_index_record = None
            self._invalidate_cached_engines()

    def delete(self) -> None:
        self.clear()

    def migrate(self, target_store_url: str) -> AdaptiveStore:
        migrated = AdaptiveStore.open(
            target_store_url,
            build_config=self._build_config(),
            suggest_config=self._suggest_config(),
        )
        migrated.add_documents(self.list_documents())
        return migrated

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
        return self.with_suggest_config(suggest_config or self._suggest_config()).suggest(
            text,
            topk=topk,
            max_words=max_words,
            length_bias=length_bias,
            collapse_prefix_ladders=collapse_prefix_ladders,
            reranker=reranker,
        )

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
        return self.with_suggest_config(suggest_config or self._suggest_config()).inspect(
            text,
            topk=topk,
            max_words=max_words,
            length_bias=length_bias,
            collapse_prefix_ladders=collapse_prefix_ladders,
            reranker=reranker,
        )

    def warm(
        self,
        sample_query: str = "a",
        *,
        topk: int = 1,
        suggest_config: SuggestConfig | None = None,
    ) -> None:
        self.with_suggest_config(suggest_config or self._suggest_config()).warm(sample_query, topk=topk)

    def with_suggest_config(self, suggest_config: SuggestConfig) -> AdaptiveAutocomplete:
        return AdaptiveAutocomplete(self, suggest_config=suggest_config)

    def _ensure_record(self) -> StoreRecord:
        if self._record is not None:
            return self._record
        self._record = self._store.open_record(build_config=BuildConfig(), suggest_config=SuggestConfig())
        return self._record

    def _build_config(self) -> BuildConfig:
        return self._record.build_config if self._record is not None else BuildConfig()

    def _suggest_config(self) -> SuggestConfig:
        return self._record.suggest_config if self._record is not None else SuggestConfig()

    def _compiled_engine(self) -> Autocomplete | None:
        with self._lock:
            return self._compiled_engine_unlocked()

    def _compiled_engine_unlocked(self) -> Autocomplete | None:
        if self._record is None:
            return None
        if self._current_index_record is None:
            current_index = self._store.load_current_index_record()
            if current_index is None:
                current_index = self._rebuild_if_documents_exist()
                if current_index is None:
                    return None
            self._current_index_record = current_index
        current_index = self._current_index_record
        cache_key = (
            self._store.cache_key(),
            current_index.compiled_index_id,
        )
        with _ENGINE_CACHE_LOCK:
            cached_engine = _ENGINE_CACHE.get(cache_key)
        if cached_engine is None:
            compiled = self._store.load_compiled_index_binary(current_index.compiled_index_id)
            cached_engine = Autocomplete._from_compiled(
                compiled,
                build_config=current_index.build_config,
                suggest_config=current_index.suggest_config,
            )
            with _ENGINE_CACHE_LOCK:
                _ENGINE_CACHE.put(cache_key, cached_engine)
        return cached_engine

    def _rebuild_if_documents_exist(self):
        if self._record is None:
            return None
        all_docs = self._store.fetch_all_documents()
        if not all_docs.documents:
            self._store.clear_current_index()
            self._record = self._store.load_record()
            self._invalidate_cached_engines()
            return None
        t0 = time.perf_counter()
        compiled, _prefix_index, build_stats = compile_index(
            all_docs.documents,
            build_config=self._record.build_config,
            suggest_config=self._record.suggest_config,
        )
        compile_ms = int((time.perf_counter() - t0) * 1000)

        record = self._store.replace_current_index(self._record, compiled=compiled)
        self._record = self._store.load_record()
        self._invalidate_cached_engines(compiled_index_id=record.compiled_index_id)

        warmup = Autocomplete._from_compiled(
            compiled,
            build_config=record.build_config,
            suggest_config=record.suggest_config,
        )
        t1 = time.perf_counter()
        warmup.suggest("a", topk=3)
        sample_suggest_ms = int((time.perf_counter() - t1) * 1000)

        self._store.insert_build_history(
            doc_count=build_stats.doc_count,
            segment_count_pre_prune=build_stats.segment_count_pre_prune,
            token_pos_pre_prune=build_stats.token_pos_pre_prune,
            segment_count_post_prune=build_stats.segment_count_post_prune,
            token_pos_post_prune=build_stats.token_pos_post_prune,
            vocab_size=build_stats.vocab_size,
            pruned_type_count=build_stats.pruned_type_count,
            phrase_count=build_stats.phrase_count,
            compile_ms=compile_ms,
            sample_suggest_ms=sample_suggest_ms,
        )
        return record

    def _mark_serving_artifacts_stale(self) -> None:
        if self._record is not None:
            self._store.clear_current_index()
            self._record = self._store.load_record()
        self._current_index_record = None
        self._invalidate_cached_engines()

    def _invalidate_cached_engines(self, *, compiled_index_id: int | None = None) -> None:
        with _ENGINE_CACHE_LOCK:
            _ENGINE_CACHE.invalidate(
                lambda key: key[0] == self._store.cache_key()
                and (compiled_index_id is None or key[1] != compiled_index_id)
            )

    @staticmethod
    def _prepare_document(document: Document) -> Document:
        return Document(
            text=document.text,
            doc_id=document.doc_id or secrets.token_urlsafe(12),
            metadata=dict(document.metadata),
        )


class AdaptiveAutocomplete:
    def __init__(self, store: AdaptiveStore, *, suggest_config: SuggestConfig | None = None) -> None:
        self._store = store
        self._suggest_config = suggest_config or SuggestConfig()

    def suggest(
        self,
        text: str,
        *,
        topk: int | None = None,
        max_words: int | None = None,
        length_bias: float | None = None,
        collapse_prefix_ladders: bool | None = None,
        reranker: BaseReranker | None = None,
    ) -> list[str]:
        engine = self._store._compiled_engine()
        if engine is None:
            return []
        return engine.suggest(
            text,
            topk=topk,
            max_words=max_words,
            length_bias=length_bias,
            collapse_prefix_ladders=collapse_prefix_ladders,
            suggest_config=self._suggest_config,
            reranker=reranker,
        )

    def inspect(
        self,
        text: str,
        *,
        topk: int | None = None,
        max_words: int | None = None,
        length_bias: float | None = None,
        collapse_prefix_ladders: bool | None = None,
        reranker: BaseReranker | None = None,
    ):
        engine = self._store._compiled_engine()
        if engine is None:
            return []
        return engine.inspect(
            text,
            topk=topk,
            max_words=max_words,
            length_bias=length_bias,
            collapse_prefix_ladders=collapse_prefix_ladders,
            suggest_config=self._suggest_config,
            reranker=reranker,
        )

    def warm(self, sample_query: str = "a", *, topk: int = 1) -> None:
        engine = self._store._compiled_engine()
        if engine is not None:
            engine.warm(sample_query, topk=topk)


__all__ = ["AdaptiveAutocomplete", "AdaptiveStore", "DeleteResult", "DuplicateDocument", "IngestResult"]
