from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from query_autocomplete.indexing.prefix_index import PrefixIndex

from tests.test_support import fake_marisa_trie_module


class PrefixIndexTests(unittest.TestCase):
    def test_get_block_id_returns_expected_match(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"inf": 3, "infl": 8, "gro": 2})
            self.assertEqual(index.get_block_id("inf"), 3)

    def test_get_block_id_returns_none_for_missing_prefix(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"inf": 3})
            self.assertIsNone(index.get_block_id("zzz"))

    def test_fuzzy_lookup_matches_deletion(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"build": 7})
            self.assertEqual(index.lookup_prefix_blocks("buid", fuzzy_prefix=True, max_edit_distance=1), [(7, 1, "build")])

    def test_fuzzy_lookup_matches_insertion(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"build": 7})
            self.assertEqual(index.lookup_prefix_blocks("buiild", fuzzy_prefix=True, max_edit_distance=1), [(7, 1, "build")])

    def test_fuzzy_lookup_matches_substitution(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"build": 7})
            self.assertEqual(index.lookup_prefix_blocks("builx", fuzzy_prefix=True, max_edit_distance=1), [(7, 1, "build")])

    def test_fuzzy_lookup_matches_transposition(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"build": 7})
            self.assertEqual(index.lookup_prefix_blocks("biuld", fuzzy_prefix=True, max_edit_distance=1), [(7, 1, "build")])

    def test_auto_fuzzy_ignores_very_short_prefixes(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"by": 7})
            self.assertEqual(index.lookup_prefix_blocks("bx", fuzzy_prefix="auto", max_edit_distance=1), [])

    def test_save_and_load_preserve_mapping(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            index = PrefixIndex()
            index.build({"inf": 3, "infr": 9})
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "prefix.marisa"
                index.save(path)
                restored = PrefixIndex()
                restored.load(path)
            self.assertEqual(restored.to_dict(), {"inf": 3, "infr": 9})


if __name__ == "__main__":
    unittest.main()
