from __future__ import annotations

import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from query_autocomplete import Autocomplete, Document
from query_autocomplete.binary_format import SCORER_FILENAME
from query_autocomplete.builder import _phrase_branching_score, compile_index
from query_autocomplete.config import BuildConfig, NormalizationConfig, SuggestConfig, apply_quality_profile
from query_autocomplete.reranking import BaseReranker

from tests.test_support import fake_marisa_trie_module


class ReverseReranker(BaseReranker):
    def rerank(self, prefix: str, candidates: list[str]) -> list[str]:
        return list(reversed(candidates))


class RuntimeQualityTests(unittest.TestCase):
    def test_create_accepts_raw_text_strings(self) -> None:
        with self._patched_modules():
            index = Autocomplete.create(
                [
                    "how to build a deck",
                    "how to build a desk",
                    "how to build with python",
                ]
            )
            results = index.suggest("how to bui", topk=5)
        self.assertIn("how to build", results)

    def test_longer_multi_token_continuations_work_without_mined_phrases(self) -> None:
        docs = [
            Document(text="alpha beta gamma delta"),
            Document(text="alpha beta gamma epsilon"),
            Document(text="alpha beta gamma zeta"),
        ]
        build_config = BuildConfig(phrase_min_count=99, phrase_max_len=4)
        with self._patched_modules():
            index = Autocomplete.create(
                docs,
                build_config=build_config,
                suggest_config=SuggestConfig(collapse_prefix_ladders=False),
            )
            results = index.suggest("alpha ", topk=5, max_words=4)
        self.assertIn("alpha beta gamma", results)
        self.assertIn("alpha beta gamma delta", results)

    def test_fragment_ranking_prefers_contextual_completion(self) -> None:
        docs = [Document(text="how to build a deck")] + ([Document(text="builder tools")] * 8) + ([Document(text="building blocks")] * 8)
        with self._patched_modules():
            index = Autocomplete.create(docs, suggest_config=SuggestConfig(collapse_prefix_ladders=False))
            results = index.suggest("how to bui", topk=5)
        self.assertEqual(results[0], "how to build")

    def test_fuzzy_prefix_recovers_single_transposition_typo(self) -> None:
        docs = [Document(text="how to build a deck"), Document(text="how to build a desk")]
        with self._patched_modules():
            index = Autocomplete.create(docs, suggest_config=SuggestConfig(collapse_prefix_ladders=False))
            results = index.suggest("how to biuld", topk=5)
        self.assertIn("how to build", results)

    def test_inspect_exposes_fuzzy_prefix_match(self) -> None:
        docs = [Document(text="how to build a deck"), Document(text="how to build a desk")]
        with self._patched_modules():
            index = Autocomplete.create(docs, suggest_config=SuggestConfig(collapse_prefix_ladders=False))
            exact = index.inspect("how to bui", topk=1)[0]
            fuzzy = index.inspect("how to biuld", topk=1)[0]
        assert exact.prefix_match is not None
        self.assertEqual(exact.prefix_match.fragment, "bui")
        self.assertEqual(exact.prefix_match.matched, "bui")
        self.assertEqual(exact.prefix_match.edit_distance, 0)
        self.assertFalse(exact.prefix_match.fuzzy)
        assert fuzzy.prefix_match is not None
        self.assertEqual(fuzzy.prefix_match.fragment, "biuld")
        self.assertEqual(fuzzy.prefix_match.matched, "build")
        self.assertEqual(fuzzy.prefix_match.edit_distance, 1)
        self.assertTrue(fuzzy.prefix_match.fuzzy)

    def test_fuzzy_prefix_can_be_disabled_at_runtime(self) -> None:
        docs = [Document(text="how to build a deck"), Document(text="how to build a desk")]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            results = index.suggest("how to biuld", topk=5, suggest_config=SuggestConfig(fuzzy_prefix=False))
        self.assertEqual(results, [])

    def test_auto_fuzzy_prefix_does_not_expand_very_short_fragments(self) -> None:
        docs = [Document(text="how to build a deck")]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            results = index.suggest("how to bx", topk=5)
        self.assertEqual(results, [])

    def test_phrase_branching_uses_sliding_prefixes(self) -> None:
        token_to_id = {"zero": 0, "one": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5}
        token_lines = [["zero", "alpha", "beta", "gamma"], ["one", "alpha", "beta", "delta"]]
        scores = _phrase_branching_score(token_to_id, {0: (2, 3, 4)}, token_lines)
        self.assertAlmostEqual(scores[0], math.log(3.0))

    def test_scorer_payload_is_persisted_and_restored(self) -> None:
        docs = [Document(text="how to build a deck"), Document(text="how to build a desk"), Document(text="how to build with python")]
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = Path(tmpdir) / "artifact"
                index = Autocomplete.create(docs)
                index.save(artifact_dir)
                self.assertTrue((artifact_dir / SCORER_FILENAME).exists())
                self.assertEqual(index._compiled.scorer_payload["type"], "interpolated_kneser_ney")  # type: ignore[union-attr]
                loaded = Autocomplete.load(artifact_dir)
                self.assertEqual(index.suggest("how to bui", topk=5), loaded.suggest("how to bui", topk=5))
                self.assertEqual(index.suggest("how to biuld", topk=5), loaded.suggest("how to biuld", topk=5))

    def test_suggest_accepts_runtime_override_config(self) -> None:
        docs = [Document(text="alpha beta gamma"), Document(text="alpha beta gamma delta"), Document(text="alpha beta gamma epsilon")]
        with self._patched_modules():
            index = Autocomplete.create(docs, suggest_config=SuggestConfig(max_suggestion_words=1))
            self.assertEqual(index.suggest("alpha beta ", topk=3)[0], "alpha beta gamma")
            long_results = index.suggest(
                "alpha beta ",
                topk=3,
                suggest_config=SuggestConfig(max_suggestion_words=3, beam_width=12, collapse_prefix_ladders=False),
            )
        self.assertIn("alpha beta gamma delta", long_results)

    def test_create_accepts_phrase_min_count_easy_knob(self) -> None:
        docs = [Document(text="alpha beta"), Document(text="alpha gamma")]
        with self._patched_modules():
            index = Autocomplete.create(docs, quality_profile="precision", phrase_min_count=1)
        self.assertEqual(index._build_config.phrase_min_count, 1)

    def test_suggest_warns_when_max_words_exceeds_build_budget(self) -> None:
        docs = [Document(text="alpha beta gamma delta epsilon")]
        with self._patched_modules():
            index = Autocomplete.create(docs, max_generated_words=2)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                index.suggest("alpha ", max_words=3)
        self.assertTrue(any("BuildConfig.max_generated_words=2" in str(item.message) for item in caught))

    def test_suggest_warns_when_fragment_exceeds_indexed_prefix_budget(self) -> None:
        docs = [Document(text="supercalifragilistic")]
        with self._patched_modules():
            index = Autocomplete.create(docs, build_config=BuildConfig(max_indexed_prefix_chars=4))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                index.suggest("super", topk=3)
        self.assertTrue(any("max_indexed_prefix_chars=4" in str(item.message) for item in caught))

    def test_suggest_warns_when_branch_limits_exceed_built_fanout(self) -> None:
        docs = [
            Document(text="alpha beta gamma"),
            Document(text="alpha beta delta"),
            Document(text="alpha beta epsilon"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(
                docs,
                build_config=BuildConfig(top_next_tokens=1, top_next_phrases=1),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                index.suggest(
                    "alpha ",
                    suggest_config=SuggestConfig(token_branch_limit=2, phrase_branch_limit=2),
                )
        messages = [str(item.message) for item in caught]
        self.assertTrue(any("BuildConfig.top_next_tokens=1" in message for message in messages))
        self.assertTrue(any("BuildConfig.top_next_phrases=1" in message for message in messages))

    def test_suggest_and_inspect_accept_runtime_reranker(self) -> None:
        docs = [
            Document(text="alpha beta gamma"),
            Document(text="alpha beta delta"),
            Document(text="alpha beta epsilon"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(docs, suggest_config=SuggestConfig(collapse_prefix_ladders=False))
            original = index.suggest("alpha beta ", topk=3, max_words=1)
            reranked = index.suggest("alpha beta ", topk=3, max_words=1, reranker=ReverseReranker())
            diagnostics = index.inspect("alpha beta ", topk=3, max_words=1, reranker=ReverseReranker())
        self.assertEqual(reranked, list(reversed(original)))
        self.assertEqual([item.text for item in diagnostics], reranked)

    def test_noise_penalty_deprioritizes_code_like_tokens(self) -> None:
        docs = [
            Document(text="deploy stable package"),
            Document(text="deploy stable package"),
            Document(text="deploy srv.123 package"),
            Document(text="deploy srv.123 package"),
            Document(text="deploy srv.123 package"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            results = index.suggest(
                "deploy s",
                topk=3,
                max_words=1,
                suggest_config=SuggestConfig(noise_penalty_weight=1.0),
            )
        self.assertEqual(results[0], "deploy stable")

    def test_redundant_continuations_are_suppressed(self) -> None:
        docs = [
            Document(text="alpha beta alpha"),
            Document(text="alpha beta alpha"),
            Document(text="alpha beta gamma"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            results = index.suggest("alpha beta ", topk=5, suggest_config=SuggestConfig(max_suggestion_words=1))
        self.assertNotIn("alpha beta alpha", results)
        self.assertIn("alpha beta gamma", results)

    def test_low_context_support_gate_can_suppress_unseen_prefixes(self) -> None:
        docs = [
            Document(text="alpha beta gamma"),
            Document(text="alpha beta delta"),
            Document(text="omega beta gamma"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            relaxed = index.suggest("alpha beta ", topk=3)
            strict = index.suggest(
                "alpha omega ",
                topk=3,
                suggest_config=SuggestConfig(min_context_support_ratio=0.75),
            )
        self.assertTrue(relaxed)
        self.assertEqual(strict, [])

    def test_inspect_mirrors_suggest_and_exposes_score_breakdown(self) -> None:
        docs = [Document(text="how to build a deck"), Document(text="how to build a desk")]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            suggestions = index.suggest("how to bui", topk=3)
            diagnostics = index.inspect("how to bui", topk=3)
        self.assertEqual([item.text for item in diagnostics], suggestions)
        self.assertTrue(diagnostics[0].expansion_trace)
        self.assertEqual(diagnostics[0].breakdown.final_score, diagnostics[0].score)

    def test_loaded_index_preserves_inspect_behavior(self) -> None:
        docs = [Document(text="how to build a deck"), Document(text="how to build a desk")]
        with self._patched_modules():
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_dir = Path(tmpdir) / "artifact"
                index = Autocomplete.create(docs)
                index.save(artifact_dir)
                loaded = Autocomplete.load(artifact_dir)
                self.assertEqual(
                    [item.text for item in index.inspect("how to bui", topk=3)],
                    [item.text for item in loaded.inspect("how to bui", topk=3)],
                )

    def test_prefix_ladder_collapse_keeps_best_representative(self) -> None:
        docs = [
            Document(text="how to build"),
            Document(text="how to build a"),
            Document(text="how to build a deck"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(docs, suggest_config=SuggestConfig(collapse_prefix_ladders=True))
            collapsed = index.suggest("how ", topk=5, max_words=4)
            expanded = index.suggest(
                "how ",
                topk=5,
                max_words=4,
                collapse_prefix_ladders=False,
            )
        self.assertLess(len(collapsed), len(expanded))

    def test_unknown_context_strategy_skip_recovers_after_noisy_token(self) -> None:
        docs = [Document(text="alpha beta gamma"), Document(text="alpha beta delta")]
        with self._patched_modules():
            index = Autocomplete.create(docs)
            strict = index.suggest(
                "alpha beta unknown ",
                topk=3,
                suggest_config=SuggestConfig(unknown_context_strategy="strict", max_suggestion_words=1),
            )
            skipped = index.suggest(
                "alpha beta unknown ",
                topk=3,
                suggest_config=SuggestConfig(unknown_context_strategy="skip", max_suggestion_words=1),
            )
        self.assertNotEqual(strict, skipped)
        self.assertTrue(skipped)

    def test_quality_profiles_apply_different_defaults(self) -> None:
        precision_build, precision_suggest = apply_quality_profile("precision")
        recall_build, recall_suggest = apply_quality_profile("recall")
        code_build, code_suggest = apply_quality_profile("code_or_logs")
        natural_build, natural_suggest = apply_quality_profile("natural_language")
        self.assertGreater(precision_build.phrase_min_count, recall_build.phrase_min_count)
        self.assertGreater(precision_suggest.noise_penalty_weight, code_suggest.noise_penalty_weight)
        self.assertTrue(natural_suggest.collapse_prefix_ladders)
        self.assertFalse(recall_suggest.collapse_prefix_ladders)
        self.assertGreaterEqual(code_build.max_indexed_prefix_chars, BuildConfig().max_indexed_prefix_chars)

    def test_quality_profile_merges_explicit_non_default_overrides(self) -> None:
        build, suggest = apply_quality_profile(
            "precision",
            build_config=BuildConfig(max_generated_words=8),
            suggest_config=SuggestConfig(default_top_k=3),
        )
        self.assertEqual(build.max_generated_words, 8)
        self.assertEqual(suggest.default_top_k, 3)
        self.assertEqual(build.phrase_min_count, 3)
        self.assertEqual(suggest.noise_penalty_weight, 0.6)

    def test_quality_profile_respects_explicit_default_value_over_profile_default(self) -> None:
        build, suggest = apply_quality_profile(
            "precision",
            build_config=BuildConfig(phrase_min_count=2),
            suggest_config=SuggestConfig(noise_penalty_weight=0.35),
        )
        self.assertEqual(build.phrase_min_count, 2)
        self.assertEqual(suggest.noise_penalty_weight, 0.35)
        self.assertEqual(build.phrase_min_doc_freq, 2)

    def test_context_width_allows_up_to_six_tokens(self) -> None:
        self.assertEqual(BuildConfig(max_context_tokens=6).max_context_tokens, 6)

    def test_context_width_over_binary_limit_fails_at_config_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_context_tokens"):
            BuildConfig(max_context_tokens=7)

    def test_document_accepts_metadata(self) -> None:
        doc = Document(text="alpha beta", metadata={"source": "unit"})
        self.assertEqual(doc.metadata["source"], "unit")

    def test_pysbd_language_is_validated_when_enabled(self) -> None:
        with self.assertRaisesRegex((ImportError, ValueError), "pysbd|language"):
            NormalizationConfig(pysbd_language="definitely-not-a-language")

    def test_phrase_doc_frequency_rejects_single_document_repetition(self) -> None:
        docs = [Document(text="alpha beta " * 8), Document(text="alpha gamma")]
        with self._patched_modules():
            compiled, _prefix, _stats = compile_index(
                docs,
                build_config=BuildConfig(phrase_min_doc_freq=2, phrase_min_pmi=-10.0, phrase_max_dominant_extension_ratio=1.0),
                suggest_config=SuggestConfig(),
            )
        phrases = self._phrase_texts(compiled)
        self.assertNotIn("alpha beta", phrases)

    def test_phrase_boundary_filter_rejects_weak_short_boundaries(self) -> None:
        docs = [Document(text="x a alpha") for _ in range(3)]
        with self._patched_modules():
            compiled, _prefix, _stats = compile_index(
                docs,
                build_config=BuildConfig(
                    phrase_min_doc_freq=1,
                    phrase_min_pmi=-10.0,
                    phrase_boundary_generic_min_count=8,
                    phrase_max_dominant_extension_ratio=1.0,
                ),
                suggest_config=SuggestConfig(),
            )
        self.assertNotIn("a alpha", self._phrase_texts(compiled))

    def test_phrase_pmi_rejects_weak_collocations(self) -> None:
        docs = [Document(text="alpha beta alpha gamma beta delta alpha beta")]
        with self._patched_modules():
            compiled, _prefix, _stats = compile_index(
                docs,
                build_config=BuildConfig(
                    phrase_min_doc_freq=1,
                    phrase_min_pmi=10.0,
                    phrase_boundary_generic_min_count=1,
                    phrase_max_dominant_extension_ratio=1.0,
                ),
                suggest_config=SuggestConfig(),
            )
        self.assertNotIn("alpha beta", self._phrase_texts(compiled))

    def test_dominant_extension_filter_rejects_incomplete_phrase_span(self) -> None:
        docs = [Document(text="alpha beta gamma") for _ in range(5)] + [Document(text="alpha beta")]
        with self._patched_modules():
            compiled, _prefix, _stats = compile_index(
                docs,
                build_config=BuildConfig(
                    phrase_min_doc_freq=1,
                    phrase_min_pmi=-10.0,
                    phrase_boundary_generic_min_count=1,
                    phrase_max_dominant_extension_ratio=0.7,
                ),
                suggest_config=SuggestConfig(),
            )
        phrases = self._phrase_texts(compiled)
        self.assertNotIn("alpha beta", phrases)
        self.assertIn("alpha beta gamma", phrases)

    def test_prefix_ladder_collapse_can_prefer_longest_candidate(self) -> None:
        docs = [
            Document(text="how to build"),
            Document(text="how to build a deck"),
            Document(text="how to build a deck safely"),
        ]
        with self._patched_modules():
            index = Autocomplete.create(docs, max_generated_words=5)
            results = index.suggest(
                "how ",
                topk=3,
                max_words=5,
                suggest_config=SuggestConfig(
                    collapse_prefix_ladders=True,
                    collapse_prefix_ladder_strategy="prefer_longest",
                ),
            )
        self.assertEqual(results[0], "how to build a deck safely")

    @staticmethod
    def _patched_modules():
        return patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()})

    @staticmethod
    def _phrase_texts(compiled) -> set[str]:
        def token_text(token_id: int) -> str:
            entry = compiled.vocab_entries[token_id]
            start = entry.offset
            return compiled.vocab_strings[start : start + entry.length].decode("utf-8")

        return {
            " ".join(token_text(token_id) for token_id in phrase.token_ids)
            for phrase in compiled.phrase_entries
        }


if __name__ == "__main__":
    unittest.main()
