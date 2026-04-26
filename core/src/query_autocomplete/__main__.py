from __future__ import annotations

import argparse
import sys
from pathlib import Path

from query_autocomplete.config import BuildConfig, QualityProfile, SuggestConfig, apply_quality_profile
from query_autocomplete.engine import Autocomplete
from query_autocomplete.input_types import coerce_documents


_PROFILES: list[str] = ["balanced", "precision", "recall", "code_or_logs", "natural_language"]


def _cmd_build(ns: argparse.Namespace) -> int:
    build_config, _ = apply_quality_profile(ns.profile)

    overrides: dict[str, object] = {}
    if ns.max_generated_words is not None:
        overrides["max_generated_words"] = max(1, ns.max_generated_words)
    if ns.max_indexed_prefix_chars is not None:
        overrides["max_indexed_prefix_chars"] = max(1, ns.max_indexed_prefix_chars)
    if overrides:
        from dataclasses import replace
        build_config = replace(build_config, **overrides)

    docs = coerce_documents(ns.input)
    print(f"Building index from {len(docs)} file(s) with profile '{ns.profile}'...")
    engine = Autocomplete.create(docs, build_config=build_config, quality_profile=ns.profile)
    engine.save(str(ns.output))
    print(f"Index saved to: {ns.output}")
    return 0


def _cmd_suggest(ns: argparse.Namespace) -> int:
    print(f"Loading index from {ns.index}...")
    engine = Autocomplete.load(ns.index)

    suggest_kwargs: dict[str, object] = {}
    if ns.topk is not None:
        suggest_kwargs["topk"] = max(1, ns.topk)
    if ns.max_words is not None:
        suggest_kwargs["max_words"] = max(1, ns.max_words)
    if ns.length_bias is not None:
        suggest_kwargs["length_bias"] = ns.length_bias

    if ns.query:
        results = engine.suggest(ns.query, **suggest_kwargs)
        for r in results:
            print(r)
        return 0

    print("Interactive mode — type a prefix and press Enter. Empty line or Ctrl-C to quit.\n")
    try:
        while True:
            try:
                query = input("> ").strip()
            except EOFError:
                break
            if not query:
                break
            results = engine.suggest(query, **suggest_kwargs)
            if results:
                for r in results:
                    print(f"  {r}")
            else:
                print("  (no suggestions)")
    except KeyboardInterrupt:
        print()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m query_autocomplete",
        description="Build and query a prefix-autocomplete index.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── build ──────────────────────────────────────────────────────────────
    build_p = sub.add_parser("build", help="Build an index from one or more corpus files.")
    build_p.add_argument(
        "--input", type=Path, nargs="+", required=True,
        metavar="FILE", help="Corpus files (.txt, .pdf, or .docx). Multiple files are merged.",
    )
    build_p.add_argument("--output", type=Path, required=True, metavar="DIR", help="Artifact directory to create.")
    build_p.add_argument(
        "--profile", default="balanced", choices=_PROFILES, metavar="PROFILE",
        help=f"Quality profile. One of: {', '.join(_PROFILES)}. Default: balanced.",
    )
    build_p.add_argument("--max-generated-words", type=int, default=None, metavar="N",
                         help="Maximum words generated per suggestion. Default: profile default (4).")
    build_p.add_argument("--max-indexed-prefix-chars", type=int, default=None, metavar="N",
                         help="Maximum prefix length indexed. Default: profile default (24).")

    # ── suggest ────────────────────────────────────────────────────────────
    suggest_p = sub.add_parser("suggest", help="Query suggestions from a saved index.")
    suggest_p.add_argument("--index", type=Path, required=True, metavar="DIR", help="Artifact directory to load.")
    suggest_p.add_argument("--query", type=str, default=None, metavar="TEXT",
                           help="Prefix to complete. Omit to enter interactive mode.")
    suggest_p.add_argument("--topk", type=int, default=None, metavar="N", help="Number of suggestions. Default: 10.")
    suggest_p.add_argument("--max-words", type=int, default=None, metavar="N",
                           help="Maximum words in each suggestion. Default: 4.")
    suggest_p.add_argument("--length-bias", type=float, default=None, metavar="F",
                           help="Length bias in [0, 1]; higher favours longer suggestions. Default: 0.5.")

    ns = parser.parse_args(list(argv) if argv is not None else None)

    if ns.command == "build":
        return _cmd_build(ns)
    if ns.command == "suggest":
        return _cmd_suggest(ns)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
