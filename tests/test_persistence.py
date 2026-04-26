from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_autocomplete import Autocomplete, Document
from query_autocomplete.binary_format import (
    CONTEXT_GRAPH_FILENAME,
    PHRASE_LEXICON_FILENAME,
    SCORER_FILENAME,
    SERVE_SCORES_FILENAME,
    TOKEN_POSTINGS_FILENAME,
    VOCAB_BIN_FILENAME,
)
from query_autocomplete.config import BuildConfig, SuggestConfig

from tests.test_support import fake_marisa_trie_module


DOCS = [
    Document(text="How to build a deck", doc_id="1"),
    Document(text="How to build a desk", doc_id="2"),
    Document(text="How to build an api", doc_id="3"),
    Document(text="How to build your app", doc_id="4"),
    Document(text="Apple pie recipes", doc_id="5"),
]


class PersistenceTests(unittest.TestCase):
    def test_round_trip_preserves_fragment_continuation(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                engine = Autocomplete.create(DOCS, max_generated_words=4)
                before_fragment = engine.suggest("how to bui", topk=5)
                before_fuzzy = engine.suggest("how to biuld", topk=5)

                artifact_dir = Path(tmpdir) / "artifact"
                engine.save(str(artifact_dir))
                loaded = Autocomplete.load(str(artifact_dir))

                self.assertEqual(before_fragment, loaded.suggest("how to bui", topk=5))
                self.assertEqual(before_fuzzy, loaded.suggest("how to biuld", topk=5))

    def test_binary_artifact_files_are_written(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = Path(tmpdir) / "artifact"
                build_config = BuildConfig(max_generated_words=6, max_indexed_prefix_chars=12, max_context_tokens=2, top_tokens_per_prefix=20, top_next_tokens=10, top_next_phrases=5, phrase_min_count=3, phrase_max_len=3)
                suggest_config = SuggestConfig(default_top_k=7, default_length_bias=0.25, max_suggestion_words=6)
                Autocomplete.create(DOCS, build_config=build_config, suggest_config=suggest_config).save(artifact_dir)

                manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["format_version"], 2)
                self.assertTrue((artifact_dir / VOCAB_BIN_FILENAME).exists())
                self.assertTrue((artifact_dir / TOKEN_POSTINGS_FILENAME).exists())
                self.assertTrue((artifact_dir / PHRASE_LEXICON_FILENAME).exists())
                self.assertTrue((artifact_dir / CONTEXT_GRAPH_FILENAME).exists())
                self.assertTrue((artifact_dir / SERVE_SCORES_FILENAME).exists())
                self.assertTrue((artifact_dir / SCORER_FILENAME).exists())

    def test_context_graph_round_trips_six_token_history_keys(self) -> None:
        docs = [Document(text="one two three four five six seven eight")]
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = Path(tmpdir) / "artifact"
                engine = Autocomplete.create(docs, build_config=BuildConfig(max_context_tokens=6))
                assert engine._compiled is not None
                self.assertTrue(any(len(key) == 6 for key in engine._compiled.context_edges))
                engine.save(artifact_dir)
                loaded = Autocomplete.load(artifact_dir)
                assert loaded._compiled is not None
                self.assertTrue(any(len(key) == 6 for key in loaded._compiled.context_edges))

    def test_managed_and_explicit_artifact_path_resolution(self) -> None:
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                previous_cwd = Path.cwd()
                os.chdir(tmpdir)
                try:
                    default_engine = Autocomplete.create(DOCS, max_generated_words=4)
                    default_engine.save()
                    artifact_root = Path(tmpdir) / ".query_autocomplete_artifacts"
                    artifact_dirs = list(artifact_root.iterdir())
                    self.assertEqual(len(artifact_dirs), 1)
                    self.assertTrue((artifact_dirs[0] / "manifest.json").exists())
                    self.assertFalse(artifact_dirs[0].name.startswith("how-to-build"))
                    self.assertTrue(Autocomplete.load(artifact_dirs[0].name).suggest("how to bui", topk=3))
                finally:
                    os.chdir(previous_cwd)

    def test_load_rejects_unsupported_manifest_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "manifest.json").write_text(json.dumps({"format_version": 999, "files": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported artifact version"):
                Autocomplete.load(tmpdir)

    def test_load_rejects_malformed_binary_file(self) -> None:
            with self._patched_modules():
                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact_dir = Path(tmpdir) / "artifact"
                    engine = Autocomplete.create(DOCS, max_generated_words=4)
                    engine.save(str(artifact_dir))
                    (artifact_dir / TOKEN_POSTINGS_FILENAME).write_bytes(b"bad")
                    with self.assertRaisesRegex(ValueError, "Malformed token postings artifact file"):
                        Autocomplete.load(str(artifact_dir))

    def test_create_requires_documents(self) -> None:
        with self._patched_modules():
            with self.assertRaisesRegex(ValueError, "Cannot build with no documents"):
                Autocomplete.create([])

    def test_export_documents_preserves_source_documents_for_adaptive_import(self) -> None:
        with self._patched_modules():
            engine = Autocomplete.create(DOCS)
            exported = engine.export_documents()
            self.assertEqual([doc.doc_id for doc in exported], [doc.doc_id for doc in DOCS])
            self.assertEqual([doc.text for doc in exported], [doc.text for doc in DOCS])

    @staticmethod
    def _patched_modules():
        return patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()})


if __name__ == "__main__":
    unittest.main()
