from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import query_autocomplete.adaptive as adaptive_module
from query_autocomplete import AdaptiveAutocomplete, AdaptiveStore, Autocomplete, Document
from query_autocomplete.config import BuildConfig, SuggestConfig
from query_autocomplete.models import BinaryIndexData

from tests.test_support import fake_marisa_trie_module


DOCS = [
    Document(text="How to build a deck", doc_id="1"),
    Document(text="How to build a desk", doc_id="2"),
    Document(text="How to build with python", doc_id="3"),
]


class AdaptiveSqlTests(unittest.TestCase):
    def test_open_persists_build_config(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path), build_config=BuildConfig(max_generated_words=5))
                loaded = AdaptiveStore.open(str(db_path))
                self.assertEqual(store._record.build_config.max_generated_words, 5)  # type: ignore[union-attr]
                self.assertEqual(loaded._record.build_config.max_generated_words, 5)  # type: ignore[union-attr]

    def test_open_is_rerunnable_and_rejects_mismatched_explicit_config(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                AdaptiveStore.open(str(db_path), build_config=BuildConfig(max_generated_words=5))
                AdaptiveStore.open(str(db_path), build_config=BuildConfig(max_generated_words=5))
                with self.assertRaisesRegex(ValueError, "different config"):
                    AdaptiveStore.open(str(db_path), build_config=BuildConfig(max_generated_words=2))

    def test_open_or_create_alias_uses_corpusless_signature(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open_or_create(str(db_path))
                store.add_documents(["alpha beta gamma"])
                self.assertEqual([doc.text for doc in AdaptiveStore.open(str(db_path)).list_documents()], ["alpha beta gamma"])

    def test_store_suggest_returns_current_results_after_add_and_remove(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents([Document(text="alpha beta gamma", doc_id="a")])
                self.assertIn("alpha beta gamma", store.suggest("alpha beta ", topk=5))
                deleted = store.remove_document("a")
                self.assertEqual(deleted.deleted_count, 1)
                self.assertEqual(store.list_documents(), [])
                self.assertEqual(store.suggest("alpha beta ", topk=5), [])
                self.assertEqual(AdaptiveStore.open(str(db_path)).list_documents(), [])

    def test_add_documents_accepts_raw_text_strings_and_generates_doc_ids(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                result = store.add_documents(["alpha beta gamma", "alpha beta delta"])
                self.assertEqual(result.inserted_count, 2)
                listed = store.list_documents()
                self.assertEqual([doc.text for doc in listed], ["alpha beta gamma", "alpha beta delta"])
                self.assertTrue(all(doc.doc_id for doc in listed))

    def test_duplicate_doc_id_and_duplicate_text_are_reported(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents([Document(text="repeat me", doc_id="a")])
                duplicate_id = store.add_documents([Document(text="repeat me", doc_id="a")])
                duplicate_text = store.add_documents([Document(text="repeat me", doc_id="b")])
                self.assertEqual(duplicate_id.duplicate_doc_ids, ("a",))
                self.assertEqual(len(duplicate_text.duplicate_texts), 1)
                self.assertEqual(duplicate_text.duplicate_texts[0].doc_id, "a")
                self.assertEqual(duplicate_text.duplicate_texts[0].text_preview, "repeat me")
                self.assertNotIn("md5", str(duplicate_text.duplicate_texts[0]).lower())
                self.assertFalse(hasattr(duplicate_text.duplicate_texts[0], "content_md5"))

    def test_duplicate_doc_id_with_different_content_raises(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents([Document(text="repeat me", doc_id="a")])
                with self.assertRaisesRegex(ValueError, "different content"):
                    store.add_documents([Document(text="changed text", doc_id="a")])

    def test_list_documents_does_not_expose_content_hash(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents([Document(text="alpha beta gamma", doc_id="a")])
                listed = store.list_documents()
                self.assertEqual(len(listed), 1)
                self.assertFalse(hasattr(listed[0], "content_md5"))

    def test_delete_clears_documents_and_keeps_valid_empty_db(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents([Document(text="alpha beta gamma", doc_id="a")])
                store.delete()
                self.assertTrue(db_path.exists())
                reopened = AdaptiveStore.open(str(db_path))
                self.assertEqual(reopened.list_documents(), [])
                self.assertEqual(reopened.suggest("alpha beta ", topk=5), [])

    def test_clear_alias_clears_documents_and_keeps_file(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents([Document(text="alpha beta gamma", doc_id="a")])
                store.clear()
                self.assertTrue(db_path.exists())
                self.assertEqual(AdaptiveStore.open(str(db_path)).list_documents(), [])

    def test_with_suggest_config_returns_query_object_without_mutating_build_config(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                build_config = BuildConfig(max_generated_words=5)
                store = AdaptiveStore.open(str(db_path), build_config=build_config)
                store.add_documents(DOCS)
                query = store.with_suggest_config(SuggestConfig(default_top_k=1))
                self.assertIsInstance(query, AdaptiveAutocomplete)
                self.assertEqual(len(query.suggest("how to bui")), 1)
                loaded = AdaptiveStore.open(str(db_path))
                self.assertEqual(loaded._record.build_config.max_generated_words, 5)  # type: ignore[union-attr]

    def test_missing_or_stale_artifacts_rebuild_automatically_on_query(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(DOCS)
                store.suggest("how to bui", topk=5)
                store._store.clear_current_index()
                reloaded = AdaptiveStore.open(str(db_path))
                self.assertIn(
                    "how to build a deck",
                    reloaded.suggest("how to bui", topk=5, suggest_config=SuggestConfig(collapse_prefix_ladders=False)),
                )

    def test_repeated_queries_reuse_current_artifacts(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                self._clear_engine_cache()
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(DOCS)
                store.suggest("how to bui", topk=5)
                with patch.object(store._store, "load_compiled_index_binary", wraps=store._store.load_compiled_index_binary) as load_binary:
                    store.suggest("how to bui", topk=5)
                    store.suggest("how to bui", topk=5)
                    self.assertEqual(load_binary.call_count, 0)

    def test_warm_builds_and_caches_current_artifacts(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                self._clear_engine_cache()
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(DOCS)
                store.warm()
                self.assertIsNotNone(store._store.load_current_index_record())
                with patch.object(store._store, "load_compiled_index_binary", wraps=store._store.load_compiled_index_binary) as load_binary:
                    store.suggest("how to bui", topk=5)
                    self.assertEqual(load_binary.call_count, 0)

    def test_warm_is_noop_for_empty_store(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.warm()
                self.assertEqual(store.suggest("how to bui", topk=5), [])

    def test_compiled_index_is_persisted_as_sql_blobs_after_query(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(DOCS)
                store.suggest("how to bui", topk=5)
                conn = sqlite3.connect(db_path)
                try:
                    row = conn.execute(
                        "SELECT vocab_strings_blob, vocab_bin_blob, prefix_json, token_postings_blob, phrase_lexicon_blob, context_graph_blob, serve_scores_blob, scorer_json FROM compiled_indexes"
                    ).fetchone()
                finally:
                    conn.close()
                assert row is not None
                self.assertTrue(all(row[index] for index in range(len(row))))

    def test_corrupt_sql_blob_raises_value_error(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(DOCS)
                store.suggest("how to bui", topk=5)
                conn = sqlite3.connect(db_path)
                try:
                    compiled_index_id = conn.execute("SELECT current_index_id FROM store_metadata WHERE id = 1").fetchone()[0]
                    conn.execute("UPDATE compiled_indexes SET token_postings_blob = ? WHERE id = ?", (sqlite3.Binary(b"bad"), compiled_index_id))
                    conn.commit()
                finally:
                    conn.close()
                self._clear_engine_cache()
                with self.assertRaisesRegex(ValueError, "Malformed SQL compiled index"):
                    AdaptiveStore.open(str(db_path)).suggest("how to bui", topk=5)

    def test_missing_sql_blob_raises_value_error(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(DOCS)
                store.suggest("how to bui", topk=5)
                conn = sqlite3.connect(db_path)
                try:
                    compiled_index_id = conn.execute("SELECT current_index_id FROM store_metadata WHERE id = 1").fetchone()[0]
                    conn.execute("UPDATE compiled_indexes SET vocab_bin_blob = ? WHERE id = ?", (sqlite3.Binary(b""), compiled_index_id))
                    conn.commit()
                finally:
                    conn.close()
                self._clear_engine_cache()
                with self.assertRaisesRegex(ValueError, "Malformed SQL compiled index"):
                    AdaptiveStore.open(str(db_path)).suggest("how to bui", topk=5)

    def test_open_accepts_sqlite_url(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store_url = f"sqlite:///{db_path}"
                store = AdaptiveStore.open(store_url, build_config=BuildConfig(max_generated_words=5))
                store.add_documents(DOCS)
                loaded = AdaptiveStore.open(store_url)
                self.assertEqual(loaded._record.build_config.max_generated_words, 5)  # type: ignore[union-attr]
                self.assertEqual(store.suggest("how to bui", topk=5), loaded.suggest("how to bui", topk=5))

    def test_sqlite_three_slash_url_is_relative_to_cwd(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                previous_cwd = Path.cwd()
                os.chdir(tmpdir)
                try:
                    AdaptiveStore.open("sqlite:///relative_from_cwd.sqlite3")
                    self.assertTrue((Path(tmpdir) / "relative_from_cwd.sqlite3").exists())
                finally:
                    os.chdir(previous_cwd)

    def test_old_corpus_schema_is_rejected(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("CREATE TABLE corpora (id INTEGER PRIMARY KEY, corpus_name TEXT NOT NULL)")
                    conn.commit()
                finally:
                    conn.close()
                with self.assertRaisesRegex(ValueError, "old corpus-based schema"):
                    AdaptiveStore.open(str(db_path))

    def test_import_autocomplete_from_in_memory_engine(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                store = AdaptiveStore.open(str(db_path))
                store.add_documents(["how to build a deck", "how to build a desk"])
                self.assertIn(
                    "how to build a deck",
                    [
                        s.lower()
                        for s in store.suggest(
                            "how to bui",
                            topk=5,
                            suggest_config=SuggestConfig(collapse_prefix_ladders=False),
                        )
                    ],
                )
                engine = Autocomplete.create(DOCS)
                promoted = AdaptiveStore.import_autocomplete(str(Path(tmpdir) / "promoted.sqlite3"), engine=engine)
                self.assertEqual(engine.suggest("how to bui", topk=5), promoted.suggest("how to bui", topk=5))

    def test_promoting_loaded_artifact_without_source_documents_fails_cleanly(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = Path(tmpdir) / "artifact"
                db_path = Path(tmpdir) / "adaptive.sqlite3"
                Autocomplete.create(DOCS).save(artifact_dir)
                loaded = Autocomplete.load(artifact_dir)
                with self.assertRaisesRegex(ValueError, "no source documents are attached"):
                    AdaptiveStore.import_autocomplete(str(db_path), engine=loaded)

    def test_engine_cache_evicts_by_count_and_approximate_bytes(self) -> None:
        cache = adaptive_module._EngineCache(max_size=2, max_bytes=10_000)
        cache.put(("store", 1, 0), self._fake_engine(30))
        cache.put(("store", 2, 0), self._fake_engine(30))
        self.assertEqual(len(cache), 2)
        cache.put(("store", 3, 0), self._fake_engine(30))
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get(("store", 1, 0)))
        small = self._fake_engine(30)
        large = self._fake_engine(200)
        small_size = adaptive_module._compiled_index_approx_bytes(small)
        large_size = adaptive_module._compiled_index_approx_bytes(large)
        byte_cache = adaptive_module._EngineCache(max_size=4, max_bytes=small_size + 1)
        byte_cache.put(("store", 1, 0), small)
        byte_cache.put(("store", 2, 0), large)
        self.assertLessEqual(byte_cache.byte_size, small_size + 1)
        self.assertTrue(large_size > small_size)

    @staticmethod
    def _patched_modules():
        return patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()})

    @staticmethod
    def _clear_engine_cache() -> None:
        cache = getattr(adaptive_module, "_ENGINE_CACHE", None)
        if hasattr(cache, "clear"):
            cache.clear()

    @staticmethod
    def _fake_engine(vocab_bytes: int):
        compiled = BinaryIndexData(
            vocab_strings=b"x" * vocab_bytes,
            vocab_entries=[],
            serve_scores=[],
            token_postings=[],
            phrase_entries=[],
            context_edges={},
            prefix_to_block={},
        )
        return types.SimpleNamespace(_compiled=compiled)


if __name__ == "__main__":
    unittest.main()
