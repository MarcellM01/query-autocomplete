"""Reproducible local benchmark for query-autocomplete on WikiText-2.

The benchmark builds in-memory indexes from deterministic word-count slices of
``Salesforce/wikitext`` / ``wikitext-2-raw-v1`` and measures:

* index build time
* time to first suggestion after build
* steady-state suggestion latency across many generated prompts

From the repository root:

    python -m pip install -e ./python-package
    python -m pip install datasets
    python benchmarks/bench.py --format markdown

By default, text is read from ``benchmarks/assets/wikitext-2-raw-v1/all.txt``.
If that file is missing, the benchmark downloads WikiText-2 from Hugging Face
and writes the merged train/validation/test splits to that path.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

_BENCH = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH.parent
_CORE_SRC = _REPO_ROOT / "core" / "src"
for _path in (str(_BENCH), str(_CORE_SRC)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

from query_autocomplete import Autocomplete, Document

import get_wikitext_data as wk  # noqa: E402

_BENCH_WIKI_DIR = wk.DEFAULT_WIKITEXT_DATA_DIR
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_DEFAULT_TIERS: tuple[tuple[str, int], ...] = (
    ("S", 10_000),
    ("M", 50_000),
    ("L", 250_000),
    ("XL", 1_000_000),
)


@dataclass(frozen=True)
class CorpusSlice:
    label: str
    target_words: int
    actual_words: int
    lines: int
    text: str


@dataclass(frozen=True)
class BenchmarkRow:
    corpus: str
    target_words: int
    actual_words: int
    source_lines: int
    prompts: int
    runs: int
    build_mean_s: float
    build_min_s: float
    first_suggest_mean_ms: float
    first_suggest_min_ms: float
    suggest_mean_ms: float
    suggest_median_ms: float
    suggest_p95_ms: float


def _ensure_wikitext_files(data_dir: Path) -> int:
    if wk.wikitext_all_txt_exists(data_dir):
        return 0
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"WikiText cache {wk.ALL_TXT!r} not found in {data_dir}\n"
        "Downloading Salesforce/wikitext wikitext-2-raw-v1 from Hugging Face (one time) ...",
        file=sys.stderr,
    )
    try:
        wk.download_wikitext_all_txt(data_dir)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Wrote WikiText cache under {data_dir}", file=sys.stderr)
    return 0


def _word_count(line: str) -> int:
    return len(_WORD_RE.findall(line))


def _slice_corpus_by_words(lines: list[str], label: str, target_words: int) -> CorpusSlice:
    if target_words < 1:
        raise ValueError("target_words must be >= 1")
    selected: list[str] = []
    total_words = 0
    for line in lines:
        line_words = _word_count(line)
        if not selected and line_words == 0:
            continue
        selected.append(line)
        total_words += line_words
        if total_words >= target_words:
            break
    if not selected:
        raise ValueError("WikiText corpus is empty.")
    return CorpusSlice(
        label=label,
        target_words=target_words,
        actual_words=total_words,
        lines=len(selected),
        text="\n".join(selected),
    )


def _parse_tiers(raw: list[str] | None) -> list[tuple[str, int]]:
    if not raw:
        return list(_DEFAULT_TIERS)
    tiers: list[tuple[str, int]] = []
    for item in raw:
        if "=" in item:
            label, value = item.split("=", 1)
        else:
            label, value = f"T{len(tiers) + 1}", item
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid tier label in {item!r}")
        try:
            words = int(value.replace("_", "").replace(",", ""))
        except ValueError as e:
            raise ValueError(f"Invalid tier word count in {item!r}") from e
        if words < 1:
            raise ValueError(f"Tier word count must be >= 1 in {item!r}")
        tiers.append((label, words))
    return tiers


def _tokens(line: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(line)]


def _partial_token(token: str) -> str:
    if len(token) <= 2:
        return token[:1]
    return token[: max(2, min(len(token) - 1, math.ceil(len(token) * 0.6)))]


def _suggest_queries_from_corpus(corpus: str, *, limit: int) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    lines = [line for line in corpus.splitlines() if _word_count(line) >= 4]
    if not lines:
        return ["the "]

    stride = max(1, len(lines) // max(1, limit * 3))
    for line in lines[::stride]:
        words = _tokens(line)
        if len(words) < 4:
            continue
        for context_words in (1, 2, 3):
            if len(words) <= context_words:
                continue
            query = " ".join([*words[:context_words], _partial_token(words[context_words])])
            if query not in seen:
                seen.add(query)
                candidates.append(query)
            if len(candidates) >= limit:
                return candidates
    return candidates or ["the "]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return ordered[int(k)]
    return ordered[lower] * (upper - k) + ordered[upper] * (k - lower)


def _time_one_tier(
    corpus: CorpusSlice,
    *,
    query_count: int,
    suggest_repeats: int,
    runs: int,
    topk: int,
    max_words: int | None,
    progress: bool,
) -> BenchmarkRow:
    queries = _suggest_queries_from_corpus(corpus.text, limit=query_count)
    build_times_s: list[float] = []
    first_suggest_ms: list[float] = []
    steady_suggest_ms: list[float] = []

    for run_idx in range(runs):
        if progress:
            print(
                f"[{corpus.label}] run {run_idx + 1}/{runs}: building "
                f"{corpus.actual_words:,} words from {corpus.lines:,} lines ...",
                file=sys.stderr,
                flush=True,
            )
        t0 = time.perf_counter()
        engine = Autocomplete.create(
            [Document(text=corpus.text, doc_id=f"wikitext-{corpus.label}")],
            max_generated_words=max_words,
        )
        build_s = time.perf_counter() - t0
        build_times_s.append(build_s)
        if progress:
            print(f"[{corpus.label}] run {run_idx + 1}/{runs}: build done in {build_s:.3f}s", file=sys.stderr, flush=True)

        first_query = queries[run_idx % len(queries)]
        if progress:
            print(f"[{corpus.label}] run {run_idx + 1}/{runs}: measuring first suggestion ...", file=sys.stderr, flush=True)
        t1 = time.perf_counter()
        engine.suggest(first_query, topk=topk, max_words=max_words)
        first_ms = (time.perf_counter() - t1) * 1000.0
        first_suggest_ms.append(first_ms)
        if progress:
            print(f"[{corpus.label}] run {run_idx + 1}/{runs}: first suggestion {first_ms:.3f}ms", file=sys.stderr, flush=True)

        if progress:
            print(f"[{corpus.label}] run {run_idx + 1}/{runs}: warming {len(queries)} prompts ...", file=sys.stderr, flush=True)
        for query in queries:
            engine.suggest(query, topk=topk, max_words=max_words)

        total_measurements = suggest_repeats * len(queries)
        if progress:
            print(
                f"[{corpus.label}] run {run_idx + 1}/{runs}: measuring "
                f"{total_measurements:,} steady-state suggestions ...",
                file=sys.stderr,
                flush=True,
            )
        for _ in range(suggest_repeats):
            for query in queries:
                t2 = time.perf_counter()
                engine.suggest(query, topk=topk, max_words=max_words)
                steady_suggest_ms.append((time.perf_counter() - t2) * 1000.0)
        if progress:
            run_measurements = steady_suggest_ms[-total_measurements:]
            print(
                f"[{corpus.label}] run {run_idx + 1}/{runs}: steady mean "
                f"{statistics.fmean(run_measurements):.3f}ms over {total_measurements:,} suggestions",
                file=sys.stderr,
                flush=True,
            )

    return BenchmarkRow(
        corpus=corpus.label,
        target_words=corpus.target_words,
        actual_words=corpus.actual_words,
        source_lines=corpus.lines,
        prompts=len(queries),
        runs=runs,
        build_mean_s=statistics.fmean(build_times_s),
        build_min_s=min(build_times_s),
        first_suggest_mean_ms=statistics.fmean(first_suggest_ms),
        first_suggest_min_ms=min(first_suggest_ms),
        suggest_mean_ms=statistics.fmean(steady_suggest_ms),
        suggest_median_ms=statistics.median(steady_suggest_ms),
        suggest_p95_ms=_percentile(steady_suggest_ms, 0.95),
    )


def _format_int(value: int) -> str:
    return f"{value:,}"


def _print_table(rows: list[BenchmarkRow]) -> None:
    header = (
        f"{'corpus':<6} {'words':>11} {'lines':>8} {'build s':>9} "
        f"{'first ms':>10} {'mean ms':>9} {'p50 ms':>8} {'p95 ms':>8} {'prompts':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.corpus:<6} {_format_int(row.actual_words):>11} {_format_int(row.source_lines):>8} "
            f"{row.build_mean_s:>9.3f} {row.first_suggest_mean_ms:>10.3f} "
            f"{row.suggest_mean_ms:>9.3f} {row.suggest_median_ms:>8.3f} "
            f"{row.suggest_p95_ms:>8.3f} {row.prompts:>8}"
        )


def _print_markdown(rows: list[BenchmarkRow]) -> None:
    print("| Corpus | Words | Lines | Build mean (s) | First suggestion mean (ms) | Suggest mean (ms) | Suggest p50 (ms) | Suggest p95 (ms) | Prompts | Runs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.corpus} | {_format_int(row.actual_words)} | {_format_int(row.source_lines)} | "
            f"{row.build_mean_s:.3f} | {row.first_suggest_mean_ms:.3f} | "
            f"{row.suggest_mean_ms:.3f} | {row.suggest_median_ms:.3f} | "
            f"{row.suggest_p95_ms:.3f} | {row.prompts} | {row.runs} |"
        )


def _print_csv(rows: list[BenchmarkRow], out: TextIO) -> None:
    writer = csv.DictWriter(out, fieldnames=list(asdict(rows[0]).keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))


def _load_wikitext(args) -> tuple[list[str], str]:
    data_dir: Path = args.data_dir.resolve()
    if args.source == "hf":
        line_source = "hf"
    elif args.source == "export":
        if not wk.wikitext_all_txt_exists(data_dir):
            if _ensure_wikitext_files(data_dir) != 0:
                raise RuntimeError("Unable to create WikiText cache.")
        line_source = "export"
    else:
        if wk.wikitext_all_txt_exists(data_dir):
            line_source = "export"
        elif _ensure_wikitext_files(data_dir) == 0:
            line_source = "export"
        else:
            line_source = "hf"
    return wk.get_wikitext_data(data_dir, source=line_source), line_source


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_BENCH_WIKI_DIR,
        help="Directory containing all.txt WikiText cache (default: benchmarks/assets/wikitext-2-raw-v1)",
    )
    p.add_argument(
        "--source",
        choices=("auto", "export", "hf"),
        default="auto",
        help="auto: use cache under --data-dir, or download if missing, else in-memory HF; "
        "export: on-disk only; hf: in-memory from Hugging Face only",
    )
    p.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        metavar="LABEL=WORDS",
        help="Corpus tier by target word count. May be repeated. Default: "
        + ", ".join(f"{label}={words}" for label, words in _DEFAULT_TIERS),
    )
    p.add_argument("--queries", type=int, default=64, help="Generated prompts per corpus tier (default: 64).")
    p.add_argument("--suggest-repeats", type=int, default=5, help="Steady-state passes over generated prompts (default: 5).")
    p.add_argument("--runs", type=int, default=1, help="Fresh builds per corpus tier; use >1 to average first-suggest latency.")
    p.add_argument("--topk", type=int, default=5, help="Suggestions requested per prompt (default: 5).")
    p.add_argument("--max-words", type=int, default=4, help="Maximum generated suggestion words (default: 4).")
    p.add_argument("--format", choices=("table", "markdown", "csv", "json"), default="table")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress updates. By default progress is printed to stderr so redirected output stays clean.",
    )
    args = p.parse_args(argv)

    if args.queries < 1:
        print("--queries must be >= 1", file=sys.stderr)
        return 2
    if args.suggest_repeats < 1:
        print("--suggest-repeats must be >= 1", file=sys.stderr)
        return 2
    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 2

    try:
        tiers = _parse_tiers(args.tiers)
        all_lines, line_source = _load_wikitext(args)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"{e}", file=sys.stderr)
        print("Install `datasets` and ensure network access, or pre-fill --data-dir.", file=sys.stderr)
        return 1

    total_words = sum(_word_count(line) for line in all_lines)
    if args.format == "table":
        print("query-autocomplete local benchmark")
        print("Dataset: Salesforce/wikitext, config: wikitext-2-raw-v1, splits: train+validation+test")
        print(f"Source: {line_source} ({args.data_dir.resolve()}); available words: {_format_int(total_words)}")
        print(f"Runs: {args.runs}; prompts per tier: {args.queries}; steady repeats: {args.suggest_repeats}; topk: {args.topk}")
        print()

    rows: list[BenchmarkRow] = []
    for label, target_words in tiers:
        if target_words > total_words:
            print(
                f"Skipping {label}: target {_format_int(target_words)} words exceeds available "
                f"{_format_int(total_words)} WikiText words.",
                file=sys.stderr,
            )
            continue
        corpus = _slice_corpus_by_words(all_lines, label, target_words)
        if not args.quiet:
            print(
                f"[{label}] prepared tier: target {target_words:,} words, "
                f"actual {corpus.actual_words:,} words, {corpus.lines:,} source lines",
                file=sys.stderr,
                flush=True,
            )
        rows.append(
            _time_one_tier(
                corpus,
                query_count=args.queries,
                suggest_repeats=args.suggest_repeats,
                runs=args.runs,
                topk=args.topk,
                max_words=args.max_words,
                progress=not args.quiet,
            )
        )
        if not args.quiet:
            print(f"[{label}] complete", file=sys.stderr, flush=True)

    if not rows:
        print("No benchmark tiers were run.", file=sys.stderr)
        return 1

    if args.format == "table":
        _print_table(rows)
    elif args.format == "markdown":
        _print_markdown(rows)
    elif args.format == "csv":
        _print_csv(rows, sys.stdout)
    else:
        payload = {
            "dataset": wk.DATASET,
            "config": wk.CONFIG,
            "splits": ["train", "validation", "test"],
            "source": line_source,
            "available_words": total_words,
            "rows": [asdict(row) for row in rows],
        }
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
