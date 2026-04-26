from __future__ import annotations

import unittest

from query_autocomplete.builder import compile_index
from query_autocomplete.config import BuildConfig, SuggestConfig
from query_autocomplete.models import BinaryIndexData, Document


def _vocab_set(data: BinaryIndexData) -> set[str]:
    raw = data.vocab_strings
    return {raw[entry.offset : entry.offset + entry.length].decode("utf-8") for entry in data.vocab_entries}


class VocabPruneTests(unittest.TestCase):
    def test_small_corpus_keeps_singleton_below_threshold(self) -> None:
        docs = [Document(text="onlyonce alpha alpha beta")]
        sc = SuggestConfig()
        data, _prefix, _stats = compile_index(
            docs,
            build_config=BuildConfig(
                vocab_prune_min_total_tokens=10_000,
                vocab_prune_min_unigram_count=2,
            ),
            suggest_config=sc,
        )
        self.assertIn("onlyonce", _vocab_set(data))

    def test_prune_removes_hapax_when_corpus_at_threshold(self) -> None:
        # threshold 0 forces pruning; min count 2 drops singleton "zonly" but keeps a and b
        docs = [Document(text="a a b b zonly")]
        sc = SuggestConfig()
        data, _prefix, _stats = compile_index(
            docs,
            build_config=BuildConfig(
                vocab_prune_min_total_tokens=0,
                vocab_prune_min_unigram_count=2,
            ),
            suggest_config=sc,
        )
        v = _vocab_set(data)
        self.assertIn("a", v)
        self.assertIn("b", v)
        self.assertNotIn("zonly", v)

    def test_prune_never_runs_when_threshold_none(self) -> None:
        docs = [Document(text="a a b zonly zextra")]
        sc = SuggestConfig()
        data, _prefix, _stats = compile_index(
            docs,
            build_config=BuildConfig(
                vocab_prune_min_total_tokens=None,
                vocab_prune_min_unigram_count=2,
            ),
            suggest_config=sc,
        )
        v = _vocab_set(data)
        self.assertIn("zonly", v)
        self.assertIn("zextra", v)

    def test_segment_prune_drops_repeat_only_in_one_line(self) -> None:
        # With line gate 0, min segment 2: "noise" appears twice in one line only -> out; a/b cross two lines.
        docs = [Document(text="a a b b noise noise\na a b b")]
        sc = SuggestConfig()
        data, _prefix, _stats = compile_index(
            docs,
            build_config=BuildConfig(
                vocab_prune_min_total_tokens=0,
                vocab_prune_min_unigram_count=2,
                vocab_prune_min_segment_freq=2,
                vocab_prune_rescue_unigram=12,
                vocab_prune_line_count_to_apply_df=0,
            ),
            suggest_config=sc,
        )
        v = _vocab_set(data)
        self.assertIn("a", v)
        self.assertIn("b", v)
        self.assertNotIn("noise", v)

    def test_rescue_keeps_high_tf_single_segment(self) -> None:
        docs = [Document(text="dense " * 20 + "\nother line")]
        sc = SuggestConfig()
        data, _prefix, _stats = compile_index(
            docs,
            build_config=BuildConfig(
                vocab_prune_min_total_tokens=0,
                vocab_prune_min_unigram_count=2,
                vocab_prune_min_segment_freq=2,
                vocab_prune_rescue_unigram=12,
                vocab_prune_line_count_to_apply_df=0,
            ),
            suggest_config=sc,
        )
        v = _vocab_set(data)
        self.assertIn("dense", v)


if __name__ == "__main__":
    unittest.main()
