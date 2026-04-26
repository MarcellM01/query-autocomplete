from __future__ import annotations

import unittest

from query_autocomplete.preprocessing.preprocess import normalize_text, tokenize_text
from query_autocomplete.config import NormalizationConfig
from query_autocomplete.scoring.local_scorer import LocalNgramScorer


class NormalizationTests(unittest.TestCase):
    def test_nfkc_and_lowercase_are_applied(self) -> None:
        cfg = NormalizationConfig()
        self.assertEqual(normalize_text("Ａpple  BUILD", config=cfg), "apple build")

    def test_punctuation_stripping_keeps_hyphens(self) -> None:
        cfg = NormalizationConfig()
        self.assertEqual(normalize_text("Hello, build-it!", config=cfg), "hello build-it")

    def test_tokenizer_preserves_contractions_and_decimals(self) -> None:
        cfg = NormalizationConfig()
        self.assertEqual(tokenize_text("We're at version 3.5 today.", config=cfg), ["we're", "at", "version", "3.5", "today"])

    def test_tokenizer_preserves_email_paths_and_codeish_tokens(self) -> None:
        cfg = NormalizationConfig()
        self.assertEqual(
            tokenize_text("Email dev@example.com about src/app.py and build_runner-v2.", config=cfg),
            ["email", "dev@example.com", "about", "src/app.py", "and", "build_runner-v2"],
        )

    def test_unicode_punctuation_is_normalized(self) -> None:
        cfg = NormalizationConfig()
        self.assertEqual(normalize_text("“Build”—it’s live…", config=cfg), "build it's live")

    def test_kneser_ney_prefers_observed_context_over_frequent_unigram(self) -> None:
        scorer = LocalNgramScorer.from_lines(
            ([["alpha", "beta"]] * 2)
            + ([["omega", "gamma"]] * 20)
            + ([["theta", "gamma"]] * 20)
        )
        self.assertGreater(scorer.score("alpha beta"), scorer.score("alpha gamma"))

    def test_scorer_payload_uses_kneser_ney_schema(self) -> None:
        scorer = LocalNgramScorer.from_lines([["alpha", "beta"]])
        payload = scorer.to_payload()
        self.assertEqual(payload["type"], "interpolated_kneser_ney")
        self.assertEqual(payload["version"], 1)
        restored = LocalNgramScorer.from_payload(payload)
        self.assertEqual(restored.unigrams, scorer.unigrams)


if __name__ == "__main__":
    unittest.main()
