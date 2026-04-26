from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from query_autocomplete.artifacts import resolve_artifact_directory
from query_autocomplete.binary_format import (
    CONTEXT_GRAPH_FILENAME,
    MANIFEST_FILENAME,
    PHRASE_LEXICON_FILENAME,
    PREFIX_FILENAME,
    SCORER_FILENAME,
    SERVE_SCORES_FILENAME,
    TOKEN_POSTINGS_FILENAME,
    VOCAB_BIN_FILENAME,
    VOCAB_STRINGS_FILENAME,
    build_manifest,
    write_context_graph,
    write_manifest,
    write_phrase_lexicon,
    write_scorer_payload,
    write_serve_scores,
    write_token_postings,
    write_vocab_bin,
    write_vocab_strings,
)
from query_autocomplete.config import BuildConfig, SuggestConfig
from query_autocomplete.indexing.prefix_index import PrefixIndex
from query_autocomplete.models import (
    BinaryIndexData,
    Document,
    EDGE_TYPE_PHRASE,
    EDGE_TYPE_TOKEN,
    TOKEN_FLAG_PHRASE_HEAD,
    TOKEN_FLAG_RARE,
    ContextEdge,
    PhraseEntry,
    TokenPosting,
    VocabEntry,
)
from query_autocomplete.preprocessing.preprocess import preprocess_training_token_lines
from query_autocomplete.scoring.local_scorer import LocalNgramScorer

@dataclass(frozen=True)
class BuildStats:
    doc_count: int
    segment_count_pre_prune: int
    token_pos_pre_prune: int
    segment_count_post_prune: int
    token_pos_post_prune: int
    vocab_size: int
    pruned_type_count: int
    phrase_count: int


class ScoreTable:
    def __init__(self) -> None:
        self._ids: dict[int, int] = {}
        self._scores: list[int] = []

    def intern(self, score: int) -> int:
        score = max(-32768, min(32767, int(score)))
        existing = self._ids.get(score)
        if existing is not None:
            return existing
        score_id = len(self._scores)
        self._scores.append(score)
        self._ids[score] = score_id
        return score_id

    @property
    def scores(self) -> list[int]:
        return list(self._scores)


def build_artifact(
    documents: list[Document],
    *,
    build_config: BuildConfig,
    suggest_config: SuggestConfig,
    output_dir: str | Path,
) -> None:
    compiled, prefix_index, _stats = compile_index(documents, build_config=build_config, suggest_config=suggest_config)
    write_compiled_index(compiled, prefix_index=prefix_index, output_dir=output_dir)


def write_compiled_index(
    compiled: BinaryIndexData,
    *,
    prefix_index: PrefixIndex,
    output_dir: str | Path,
) -> None:
    artifact_dir = resolve_artifact_directory(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    write_vocab_strings(artifact_dir / VOCAB_STRINGS_FILENAME, compiled.vocab_strings)
    write_vocab_bin(artifact_dir / VOCAB_BIN_FILENAME, compiled.vocab_entries)
    prefix_index.save(artifact_dir / PREFIX_FILENAME)
    write_token_postings(artifact_dir / TOKEN_POSTINGS_FILENAME, compiled.token_postings)
    write_phrase_lexicon(artifact_dir / PHRASE_LEXICON_FILENAME, compiled.phrase_entries)
    write_context_graph(artifact_dir / CONTEXT_GRAPH_FILENAME, compiled.context_edges)
    write_serve_scores(artifact_dir / SERVE_SCORES_FILENAME, compiled.serve_scores)
    write_scorer_payload(artifact_dir / SCORER_FILENAME, compiled.scorer_payload)
    write_manifest(artifact_dir / MANIFEST_FILENAME, compiled.metadata)


def compile_index(
    documents: list[Document],
    *,
    build_config: BuildConfig,
    suggest_config: SuggestConfig,
) -> tuple[BinaryIndexData, PrefixIndex, BuildStats]:
    if not documents:
        raise ValueError("Cannot build with no documents; provide at least one non-empty source.")

    config = build_config
    token_lines, line_doc_ids = _preprocess_training_token_lines_with_doc_ids(documents, config=config)
    if not token_lines:
        raise ValueError("Cannot build with no usable tokens after preprocessing.")

    seg_pre = len(token_lines)
    tok_pre = sum(len(l) for l in token_lines)
    pre_prune_types = len({t for line in token_lines for t in line})

    token_lines, line_doc_ids = _apply_vocab_prune_with_doc_ids(token_lines, line_doc_ids, config=config)
    if not token_lines:
        raise ValueError("Cannot build with no usable tokens after vocabulary pruning.")

    seg_post = len(token_lines)
    tok_post = sum(len(l) for l in token_lines)

    vocab, token_to_id, unigram_counts, doc_freq = _build_vocab(token_lines)
    phrase_entries, phrase_counts, phrase_lookup = _mine_phrases(
        token_lines,
        unigram_counts,
        token_to_id,
        vocab,
        config=config,
    )
    scorer_payload = LocalNgramScorer.from_lines(token_lines).to_payload()
    score_table = ScoreTable()
    vocab_strings, vocab_entries = _build_vocab_files(vocab, unigram_counts, doc_freq, phrase_entries, config=config)
    prefix_blocks, prefix_to_block = _build_prefix_blocks(
        vocab,
        token_to_id,
        unigram_counts,
        doc_freq,
        token_lines,
        score_table=score_table,
        config=config,
    )
    context_edges = _build_context_graph(
        token_lines,
        phrase_lookup,
        vocab,
        unigram_counts,
        phrase_counts,
        score_table=score_table,
        config=config,
    )

    prefix_index = PrefixIndex()
    prefix_index.build(prefix_to_block)
    manifest = build_manifest(
        config={"build": asdict(build_config), "suggest": asdict(suggest_config)},
        metadata={
            "vocab_size": len(vocab),
            "phrase_count": len(phrase_entries),
            "prefix_count": len(prefix_to_block),
            "context_count": len(context_edges),
            "files": [
                MANIFEST_FILENAME,
                VOCAB_STRINGS_FILENAME,
                VOCAB_BIN_FILENAME,
                PREFIX_FILENAME,
                TOKEN_POSTINGS_FILENAME,
                PHRASE_LEXICON_FILENAME,
                CONTEXT_GRAPH_FILENAME,
                SERVE_SCORES_FILENAME,
                SCORER_FILENAME,
            ],
            "single_shard": True,
        },
    )
    stats = BuildStats(
        doc_count=len(documents),
        segment_count_pre_prune=seg_pre,
        token_pos_pre_prune=tok_pre,
        segment_count_post_prune=seg_post,
        token_pos_post_prune=tok_post,
        vocab_size=len(vocab),
        pruned_type_count=pre_prune_types - len(vocab),
        phrase_count=len(phrase_entries),
    )
    return (
        BinaryIndexData(
            vocab_strings=vocab_strings,
            vocab_entries=vocab_entries,
            serve_scores=score_table.scores,
            token_postings=prefix_blocks,
            phrase_entries=phrase_entries,
            context_edges=context_edges,
            prefix_to_block=prefix_to_block,
            scorer_payload=scorer_payload,
            metadata=manifest,
        ),
        prefix_index,
        stats,
    )


def _apply_vocab_prune(token_lines: list[list[str]], *, config: BuildConfig) -> list[list[str]]:
    return _apply_vocab_prune_with_doc_ids(token_lines, list(range(len(token_lines))), config=config)[0]


def _apply_vocab_prune_with_doc_ids(
    token_lines: list[list[str]],
    line_doc_ids: list[int],
    *,
    config: BuildConfig,
) -> tuple[list[list[str]], list[int]]:
    if config.vocab_prune_min_total_tokens is None:
        return token_lines, line_doc_ids
    total_positions = sum(len(line) for line in token_lines)
    if total_positions < config.vocab_prune_min_total_tokens:
        return token_lines, line_doc_ids
    unigram_counts: Counter[str] = Counter()
    for line in token_lines:
        unigram_counts.update(line)
    min_c = int(config.vocab_prune_min_unigram_count)
    min_seg = int(config.vocab_prune_min_segment_freq)
    rescue = int(config.vocab_prune_rescue_unigram)
    line_gate = int(config.vocab_prune_line_count_to_apply_df)
    seg_freq: Counter[str] = Counter()
    for line in token_lines:
        for t in set(line):
            seg_freq[t] += 1
    n_lines = len(token_lines)
    use_segment = min_seg > 1 and (line_gate == 0 or n_lines >= line_gate)
    if min_c <= 1 and (not use_segment or min_seg <= 1):
        return token_lines, line_doc_ids
    if min_c > 1 and (not use_segment or min_seg <= 1):
        return _prune_by_unigram_only_with_doc_ids(token_lines, line_doc_ids, unigram_counts, min_c=min_c)

    def _keep(t: str) -> bool:
        c = unigram_counts[t]
        if c < min_c:
            return False
        if not use_segment or min_seg <= 1:
            return True
        if rescue > 0 and c >= rescue:
            return True
        return seg_freq[t] >= min_seg

    return _prune_by_predicate_with_doc_ids(token_lines, line_doc_ids, _keep)


def _preprocess_training_token_lines_with_doc_ids(
    documents: list[Document],
    *,
    config: BuildConfig,
) -> tuple[list[list[str]], list[int]]:
    token_lines: list[list[str]] = []
    line_doc_ids: list[int] = []
    for doc_index, document in enumerate(documents):
        lines = preprocess_training_token_lines([document], config=config.normalization)
        token_lines.extend(lines)
        line_doc_ids.extend([doc_index] * len(lines))
    return token_lines, line_doc_ids


def _prune_by_unigram_only(
    token_lines: list[list[str]], unigram_counts: Counter[str], *, min_c: int
) -> list[list[str]]:
    return _prune_by_unigram_only_with_doc_ids(
        token_lines, list(range(len(token_lines))), unigram_counts, min_c=min_c
    )[0]


def _prune_by_unigram_only_with_doc_ids(
    token_lines: list[list[str]], line_doc_ids: list[int], unigram_counts: Counter[str], *, min_c: int
) -> tuple[list[list[str]], list[int]]:
    out: list[list[str]] = []
    out_doc_ids: list[int] = []
    for line, doc_id in zip(token_lines, line_doc_ids):
        kept = [token for token in line if unigram_counts[token] >= min_c]
        if kept:
            out.append(kept)
            out_doc_ids.append(doc_id)
    return out, out_doc_ids


def _prune_by_predicate(
    token_lines: list[list[str]], keep: Callable[[str], bool]
) -> list[list[str]]:
    return _prune_by_predicate_with_doc_ids(token_lines, list(range(len(token_lines))), keep)[0]


def _prune_by_predicate_with_doc_ids(
    token_lines: list[list[str]], line_doc_ids: list[int], keep: Callable[[str], bool]
) -> tuple[list[list[str]], list[int]]:
    out: list[list[str]] = []
    out_doc_ids: list[int] = []
    for line, doc_id in zip(token_lines, line_doc_ids):
        kept = [token for token in line if keep(token)]
        if kept:
            out.append(kept)
            out_doc_ids.append(doc_id)
    return out, out_doc_ids


def _build_vocab(
    token_lines: list[list[str]],
) -> tuple[list[str], dict[str, int], Counter[str], Counter[str]]:
    unigram_counts: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    for tokens in token_lines:
        unigram_counts.update(tokens)
        doc_freq.update(set(tokens))
    vocab = sorted(unigram_counts, key=lambda token: (-unigram_counts[token], token))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    return vocab, token_to_id, unigram_counts, doc_freq


def _mine_phrases(
    token_lines: list[list[str]],
    unigram_counts: Counter[str],
    token_to_id: dict[str, int],
    vocab: list[str],
    *,
    config: BuildConfig,
) -> tuple[list[PhraseEntry], Counter[tuple[int, ...]], dict[tuple[int, ...], int]]:
    counts: Counter[tuple[int, ...]] = Counter()
    seg_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)
    left_extensions: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    right_extensions: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    max_len = max(2, int(config.phrase_max_len))
    for line_index, tokens in enumerate(token_lines):
        ids = [token_to_id[token] for token in tokens]
        seen_in_line: set[tuple[int, ...]] = set()
        for start in range(len(ids)):
            for width in range(2, min(max_len, len(ids) - start) + 1):
                seq = tuple(ids[start : start + width])
                counts[seq] += 1
                seen_in_line.add(seq)
                if start > 0:
                    left_extensions[seq][ids[start - 1]] += 1
                if start + width < len(ids):
                    right_extensions[seq][ids[start + width]] += 1
        for seq in seen_in_line:
            seg_sets[seq].add(line_index)
    doc_freq: Counter[tuple[int, ...]] = Counter({seq: len(segs) for seq, segs in seg_sets.items()})
    total_positions = max(1, sum(len(line) for line in token_lines))
    min_doc_freq = max(1, int(config.phrase_min_doc_freq))
    kept = [
        seq
        for seq, count in counts.items()
        if count >= config.phrase_min_count
        and doc_freq[seq] >= min_doc_freq
        and _phrase_quality(seq, counts[seq]) >= 2.0
        and _phrase_pmi(seq, count, vocab, unigram_counts, total_positions) >= float(config.phrase_min_pmi)
        and _phrase_boundary_quality(seq, count, vocab, min_count=int(config.phrase_boundary_generic_min_count))
        and not _has_dominant_extension(
            seq,
            count,
            left_extensions,
            right_extensions,
            max_ratio=float(config.phrase_max_dominant_extension_ratio),
        )
    ]
    kept.sort(key=lambda seq: (-counts[seq], len(seq), seq))
    phrases: list[PhraseEntry] = []
    lookup: dict[tuple[int, ...], int] = {}
    for phrase_id, seq in enumerate(kept):
        lookup[seq] = phrase_id
        phrases.append(
            PhraseEntry(
                phrase_id=phrase_id,
                token_ids=seq,
                display_length_chars=max(0, sum(len(vocab[token_id]) for token_id in seq) + len(seq) - 1),
                flags=0,
            )
        )
    return phrases, Counter({seq: counts[seq] for seq in kept}), lookup


def _phrase_quality(seq: tuple[int, ...], count: int) -> float:
    return math.log(count + 1.0) * len(seq)


def _phrase_pmi(
    seq: tuple[int, ...],
    count: int,
    vocab: list[str],
    unigram_counts: Counter[str],
    total_positions: int,
) -> float:
    if not seq or count <= 0:
        return float("-inf")
    phrase_probability = count / total_positions
    token_probability = 1.0
    for token_id in seq:
        token_count = unigram_counts.get(vocab[token_id], 0)
        if token_count <= 0:
            return 0.0
        token_probability *= token_count / total_positions
    if token_probability <= 0:
        return 0.0
    return math.log(phrase_probability / token_probability)


def _phrase_boundary_quality(seq: tuple[int, ...], count: int, vocab: list[str], *, min_count: int) -> bool:
    if not seq:
        return False
    left = vocab[seq[0]]
    right = vocab[seq[-1]]
    if _is_weak_boundary_token(left) and count < min_count:
        return False
    if _is_weak_boundary_token(right) and count < min_count:
        return False
    return True


def _is_weak_boundary_token(token: str) -> bool:
    return token.isalpha() and len(token) <= 2


def _has_dominant_extension(
    seq: tuple[int, ...],
    count: int,
    left_extensions: dict[tuple[int, ...], Counter[int]],
    right_extensions: dict[tuple[int, ...], Counter[int]],
    *,
    max_ratio: float,
) -> bool:
    if max_ratio >= 1.0 or count <= 0:
        return False
    dominant_left = max(left_extensions.get(seq, Counter()).values(), default=0) / count
    dominant_right = max(right_extensions.get(seq, Counter()).values(), default=0) / count
    return dominant_left > max_ratio or dominant_right > max_ratio


def _build_vocab_files(
    vocab: list[str],
    unigram_counts: Counter[str],
    doc_freq: Counter[str],
    phrase_entries: list[PhraseEntry],
    *,
    config: BuildConfig,
) -> tuple[bytes, list[VocabEntry]]:
    phrase_heads = {phrase.token_ids[0] for phrase in phrase_entries if phrase.token_ids}
    blob = bytearray()
    entries: list[VocabEntry] = []
    for token_id, token in enumerate(vocab):
        encoded = token.encode("utf-8")
        flags = 0
        if unigram_counts[token] <= 1:
            flags |= TOKEN_FLAG_RARE
        # Phrase head marking lets us cheaply prioritize phrase expansion later.
        if token_id in phrase_heads:
            flags |= TOKEN_FLAG_PHRASE_HEAD
        offset = len(blob)
        blob.extend(encoded)
        entries.append(
            VocabEntry(
                offset=offset,
                length=len(encoded),
                unigram_count=int(unigram_counts[token]),
                doc_freq=int(doc_freq[token]),
                flags=flags,
            )
        )
    return bytes(blob), entries


def _build_prefix_blocks(
    vocab: list[str],
    token_to_id: dict[str, int],
    unigram_counts: Counter[str],
    doc_freq: Counter[str],
    token_lines: list[list[str]],
    *,
    score_table: ScoreTable,
    config: BuildConfig,
) -> tuple[list[list[TokenPosting]], dict[str, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    continuation_entropy = _token_continuation_entropy(token_to_id, vocab, token_lines=token_lines)
    context_specificity = _token_context_specificity(token_to_id, vocab, token_lines=token_lines)
    for token in vocab:
        limit = min(len(token), config.max_indexed_prefix_chars)
        token_id = token_to_id[token]
        for width in range(1, limit + 1):
            buckets[token[:width]].append(token_id)
    block_map: dict[str, int] = {}
    blocks: list[list[TokenPosting]] = []
    for prefix in sorted(buckets):
        ranked = sorted(
            set(buckets[prefix]),
            key=lambda token_id: (
                -_quantize_score(
                    0.45 * math.log(unigram_counts[vocab[token_id]] + 1.0)
                    + 0.20 * math.log(doc_freq[vocab[token_id]] + 1.0)
                    + 0.20 * continuation_entropy[token_id]
                    + 0.15 * context_specificity[token_id]
                ),
                token_id,
            ),
        )[: config.top_tokens_per_prefix]
        block_id = len(blocks)
        block_map[prefix] = block_id
        blocks.append(
            [
                TokenPosting(
                    token_id=token_id,
                    score_id=score_table.intern(
                        _quantize_score(
                            0.40 * math.log(unigram_counts[vocab[token_id]] + 1.0)
                            + 0.20 * math.log(doc_freq[vocab[token_id]] + 1.0)
                            + 0.20 * continuation_entropy[token_id]
                            + 0.20 * context_specificity[token_id]
                        )
                    ),
                )
                for token_id in ranked
            ]
        )
    return blocks, block_map


def _build_context_graph(
    token_lines: list[list[str]],
    phrase_lookup: dict[tuple[int, ...], int],
    vocab: list[str],
    unigram_counts: Counter[str],
    phrase_counts: Counter[tuple[int, ...]],
    *,
    score_table: ScoreTable,
    config: BuildConfig,
) -> dict[tuple[int, ...], list[ContextEdge]]:
    token_history_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    phrase_history_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    token_to_id = {token: index for index, token in enumerate(vocab)}
    phrase_by_id = {phrase_id: key for key, phrase_id in phrase_lookup.items()}
    continuation_entropy = _token_continuation_entropy(token_to_id, vocab, token_lines=token_lines)
    context_specificity = _token_context_specificity(token_to_id, vocab, token_lines=token_lines)
    phrase_support = {
        phrase_id: math.log(phrase_counts[key] + 1.0) * len(key)
        for phrase_id, key in phrase_by_id.items()
    }
    phrase_branching = _phrase_branching_score(token_to_id, phrase_by_id, token_lines)
    for tokens in token_lines:
        ids = [token_to_id[token] for token in tokens]
        for index, token_id in enumerate(ids):
            for history_width in range(1, min(config.max_context_tokens, index) + 1):
                key = tuple(ids[index - history_width : index])
                token_history_counts[key][token_id] += 1
            for history_width in range(1, min(config.max_context_tokens, index) + 1):
                key = tuple(ids[index - history_width : index])
                for width in range(2, min(config.phrase_max_len, len(ids) - index) + 1):
                    seq = tuple(ids[index : index + width])
                    phrase_id = phrase_lookup.get(seq)
                    if phrase_id is not None:
                        phrase_history_counts[key][phrase_id] += 1
    out: dict[tuple[int, ...], list[ContextEdge]] = {}
    for key in sorted(set(token_history_counts) | set(phrase_history_counts)):
        edges: list[ContextEdge] = []
        for token_id, count in token_history_counts[key].most_common(config.top_next_tokens):
            token = vocab[token_id]
            score_id = score_table.intern(
                _quantize_score(
                    0.55 * math.log(count + 1.0)
                    + 0.20 * math.log(unigram_counts[token] + 1.0)
                    + 0.15 * continuation_entropy[token_id]
                    + 0.10 * context_specificity[token_id]
                )
            )
            edges.append(ContextEdge(target_id=token_id, target_type=EDGE_TYPE_TOKEN, score_id=score_id))
        for phrase_id, count in phrase_history_counts[key].most_common(config.top_next_phrases):
            score_id = score_table.intern(
                _quantize_score(
                    0.45 * math.log(count + 1.0)
                    + 0.30 * phrase_support[phrase_id]
                    + 0.25 * phrase_branching[phrase_id]
                )
            )
            edges.append(ContextEdge(target_id=phrase_id, target_type=EDGE_TYPE_PHRASE, score_id=score_id))
        out[key] = edges
    return out


def _quantize_score(value: float) -> int:
    return int(round(value * 1024.0))


def _token_continuation_entropy(
    token_to_id: dict[str, int],
    vocab: list[str],
    *,
    token_lines: list[list[str]] | None = None,
) -> dict[int, float]:
    if token_lines is None:
        return {token_id: 0.0 for token_id in range(len(vocab))}
    next_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for tokens in token_lines:
        ids = [token_to_id[token] for token in tokens]
        for left, right in zip(ids, ids[1:]):
            next_counts[left][right] += 1
    return {token_id: _entropy(counter) for token_id, counter in next_counts.items()} | {
        token_id: 0.0 for token_id in range(len(vocab)) if token_id not in next_counts
    }


def _token_context_specificity(
    token_to_id: dict[str, int],
    vocab: list[str],
    *,
    token_lines: list[list[str]] | None = None,
) -> dict[int, float]:
    if token_lines is None:
        return {token_id: 0.0 for token_id in range(len(vocab))}
    left_contexts: dict[int, set[int]] = defaultdict(set)
    total_contexts = 0
    for tokens in token_lines:
        ids = [token_to_id[token] for token in tokens]
        for left, right in zip(ids, ids[1:]):
            left_contexts[right].add(left)
            total_contexts += 1
    total_contexts = max(total_contexts, 1)
    scores: dict[int, float] = {}
    for token_id in range(len(vocab)):
        context_count = len(left_contexts.get(token_id, set()))
        scores[token_id] = -math.log((context_count + 1.0) / (total_contexts + 1.0))
    return scores


def _phrase_branching_score(
    token_to_id: dict[str, int],
    phrase_by_id: dict[int, tuple[int, ...]],
    token_lines: list[list[str]],
) -> dict[int, float]:
    if not phrase_by_id:
        return {}
    prefix_branching: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for tokens in token_lines:
        ids = tuple(token_to_id[token] for token in tokens)
        for start in range(len(ids)):
            remaining = len(ids) - start
            for width in range(1, remaining):
                prefix_branching[ids[start : start + width]].add(ids[start + width])
    scores: dict[int, float] = {}
    for phrase_id, token_ids in phrase_by_id.items():
        prefix = token_ids[:-1]
        branch_count = len(prefix_branching.get(prefix, set()))
        scores[phrase_id] = math.log(branch_count + 1.0)
    return scores


def _entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        probability = count / total
        value -= probability * math.log(probability + 1e-12)
    return value
