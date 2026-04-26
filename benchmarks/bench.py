"""Rough local build / suggest timing: index a Wikitext slice in one :class:`Document`.

By default, text is read from ``benchmarks/assets/wikitext-2-raw-v1/all.txt``. If that file is
missing, the benchmark downloads Salesforce/wikitext-2-raw-v1 from Hugging Face
(``pip install datasets``) and writes ``all.txt`` under that path.

From the repository root:

    python -m pip install -e ./python-package
    python -m pip install datasets
    python benchmarks/bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

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

# Default three rungs are 1x / 3x / 9x this many lines (capped at corpus size). A small
# base keeps index build time down; suggest timing still shows how cost scales with data.
_BENCH_LINE_BASE = 1000


def _ensure_wikitext_files(data_dir: Path) -> int:
    if wk.wikitext_all_txt_exists(data_dir):
        return 0
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Wikitext {wk.ALL_TXT!r} not in {data_dir}\nDownloading from Hugging Face (one-time) ...",
        file=sys.stderr,
    )
    try:
        wk.download_wikitext_all_txt(data_dir)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"Wrote wikitext under {data_dir}", file=sys.stderr)
    return 0


def _suggest_queries_from_corpus(corpus: str, *, limit: int = 12) -> list[str]:
    out: list[str] = []
    for line in corpus.splitlines()[:2000]:
        words = line.split()
        if len(words) >= 2:
            out.append(f"{words[0]} {words[1]} ")
            if len(out) >= limit:
                break
    return out or ["the same "]


def _typo_query(query: str) -> str:
    stripped = query.rstrip()
    if not stripped:
        return query
    parts = stripped.split()
    token = parts[-1]
    if len(token) >= 4:
        token = token[0] + token[2] + token[1] + token[3:]
    elif len(token) >= 2:
        token = token[:-1] + "x"
    parts[-1] = token
    return " ".join(parts)


def _line_rung_targets(total: int) -> list[int]:
    if total < 1:
        return []
    b = min(_BENCH_LINE_BASE, total)
    raw = [b, 3 * b, 9 * b]
    out: list[int] = []
    for t in raw:
        t = min(max(1, t), total)
        if not out or t > out[-1]:
            out.append(t)
    return out


def _time_build(text: str) -> tuple[float, Autocomplete, list[str]]:
    t0 = time.perf_counter()
    engine = Autocomplete.create(
        [Document(text=text, doc_id="wikitext-bench")],
    )
    build_s = time.perf_counter() - t0
    return build_s, engine, _suggest_queries_from_corpus(text)


def _time_suggest(label: str, engine: Autocomplete, query: str) -> dict[str, float | str]:
    t_cold0 = time.perf_counter()
    engine.suggest(query, topk=5)
    cold_s = time.perf_counter() - t_cold0
    for _ in range(3):
        engine.suggest(query, topk=5)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        engine.suggest(query, topk=5)
    batch = time.perf_counter() - t0
    return {
        "corpus": label,
        "cold_first_suggest_s": cold_s,
        "mean_suggest_ms": (batch / n) * 1000.0,
    }


def _time_fuzzy_quality(engine: Autocomplete, queries: list[str]) -> dict[str, float]:
    typo_queries = [_typo_query(query) for query in queries]
    for query in [*queries, *typo_queries]:
        engine.suggest(query, topk=5)

    exact_n = max(1, len(queries))
    t0 = time.perf_counter()
    exact_results = [engine.suggest(query, topk=5) for query in queries]
    exact_ms = ((time.perf_counter() - t0) / exact_n) * 1000.0

    typo_n = max(1, len(typo_queries))
    t1 = time.perf_counter()
    typo_results = [engine.suggest(query, topk=5) for query in typo_queries]
    typo_ms = ((time.perf_counter() - t1) / typo_n) * 1000.0

    recovered = sum(1 for exact, typo in zip(exact_results, typo_results) if exact and typo and exact[0] in typo)
    fuzzy_seen = 0
    for query in typo_queries:
        diagnostics = engine.inspect(query, topk=5)
        if any(item.prefix_match is not None and item.prefix_match.fuzzy for item in diagnostics):
            fuzzy_seen += 1
    return {
        "exact_mean_ms": exact_ms,
        "typo_mean_ms": typo_ms,
        "top1_recovered_pct": (recovered / typo_n) * 100.0,
        "fuzzy_match_pct": (fuzzy_seen / typo_n) * 100.0,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_BENCH_WIKI_DIR,
        help="Directory containing all.txt wikitext cache (default: benchmarks/assets/wikitext-2-raw-v1)",
    )
    p.add_argument(
        "--source",
        choices=("auto", "export", "hf"),
        default="auto",
        help="auto: use cache under --data-dir, or download if missing, else in-memory HF; "
        "export: on-disk only (download if missing into --data-dir); "
        "hf: in-memory from Hugging Face only (no files)",
    )
    p.add_argument(
        "--line-rung",
        type=int,
        action="append",
        dest="line_rungs",
        metavar="N",
        help="Override line count for a benchmark step (up to 3; may repeat the flag). "
        f"Default: {_BENCH_LINE_BASE}, {3 * _BENCH_LINE_BASE}, and {9 * _BENCH_LINE_BASE} lines "
        "(1x/3x/9x, each capped to corpus size).",
    )
    args = p.parse_args(argv)

    data_dir: Path = args.data_dir.resolve()

    line_source: str
    if args.source == "hf":
        line_source = "hf"
    elif args.source == "export":
        if not wk.wikitext_all_txt_exists(data_dir):
            if _ensure_wikitext_files(data_dir) != 0:
                return 1
        line_source = "export"
    else:
        if wk.wikitext_all_txt_exists(data_dir):
            line_source = "export"
        else:
            if _ensure_wikitext_files(data_dir) == 0:
                line_source = "export"
            else:
                line_source = "hf"

    try:
        all_lines = wk.get_wikitext_data(data_dir, source=line_source)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"{e}", file=sys.stderr)
        print("Install `datasets` and ensure network access, or pre-fill --data-dir.", file=sys.stderr)
        return 1

    n_total = len(all_lines)
    if args.line_rungs:
        rungs = [min(int(x), n_total) for x in args.line_rungs[:3]]
        for r in rungs:
            if r < 1:
                print("--line-rung must be >= 1", file=sys.stderr)
                return 1
    else:
        rungs = _line_rung_targets(n_total)

    w_table = 22
    print("query-autocomplete - local benchmark (Wikitext, this machine, no isolation)")
    print(f"Data dir: {data_dir}  (lines: {n_total})  read mode: {line_source}")
    print(
        f"{'corpus':<{w_table}} {'build (s)':>12} {'1st suggest (s)':>16} "
        f"{'mean suggest (ms)':>18} {'typo ms':>10} {'recover %':>10}"
    )
    print("-" * (w_table + 12 + 16 + 18 + 10 + 10 + 5))

    for line_cap in rungs:
        if line_cap < 1:
            continue
        text = "\n".join(all_lines[:line_cap])
        label = f"first {line_cap} lines"
        build_s, engine, queries = _time_build(text)
        q = queries[0]
        srow = _time_suggest(label, engine, q)
        fuzzy = _time_fuzzy_quality(engine, queries)
        srow["build_s"] = build_s
        print(
            f"{str(srow['corpus']):<{w_table}} {build_s:>12.3f} "
            f"{srow['cold_first_suggest_s']:>16.4f} {srow['mean_suggest_ms']:>18.3f} "
            f"{fuzzy['typo_mean_ms']:>10.3f} {fuzzy['top1_recovered_pct']:>10.1f}"
        )
        print(f"  (suggest query: {q!r}; typo query: {_typo_query(q)!r}; fuzzy match rate: {fuzzy['fuzzy_match_pct']:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
