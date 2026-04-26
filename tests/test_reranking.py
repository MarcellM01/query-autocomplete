from __future__ import annotations

import unittest

from query_autocomplete.reranking import BaseReranker, HeuristicReranker
from query_autocomplete.reranking.heuristic import _after_prefix, _heuristic_key


class HeuristicRerankerTests(unittest.TestCase):
    def test_base_pass_through(self) -> None:
        c = ["b", "a", "c"]
        self.assertEqual(BaseReranker().rerank("x", list(c)), c)

    def test_after_prefix_trivial(self) -> None:
        self.assertEqual(_after_prefix("what is ", "what is the"), "the")

    def test_heuristic_prefers_shorter_or_cleaner_tails(self) -> None:
        r = HeuristicReranker()
        p = "what is the capital of france "
        # Long noisy tail should sort after a shorter, cleaner one when heuristics differ.
        noise = f"{p}and something very long " * 3 + "zzzz"
        short = f"{p}in short"
        out = r.rerank(p, [noise, short])
        self.assertEqual(out[0], short)

    def test_key_stable_ordering(self) -> None:
        a = _heuristic_key("x", "x a", 0)
        b = _heuristic_key("x", "x a", 1)
        # Same candidate shape → same first component; second preserves index
        self.assertEqual(a[0], b[0])
        self.assertLess(a[1], b[1])


if __name__ == "__main__":
    unittest.main()
