from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from query_autocomplete.config import SuggestConfig


@dataclass(frozen=True)
class BeamState:
    token_seq: tuple[int, ...]
    prior_score: float


def _remember_candidate(
    candidates: dict[tuple[int, ...], float],
    token_seq: tuple[int, ...],
    prior_score: float,
) -> None:
    existing = candidates.get(token_seq)
    if existing is None or prior_score > existing:
        candidates[token_seq] = prior_score


def beam_generate(
    history_tokens: list[str],
    history_ids: list[int | None],
    *,
    initial_beam: list[tuple[tuple[int, ...], float]],
    max_words: int,
    prefer_long: float,
    suggest_config: SuggestConfig,
    seed_with_empty: bool,
    beam_rank: Callable[[list[str], tuple[int, ...], float], float],
    best_history_key: Callable[[list[int | None] | list[int]], tuple[int, ...]],
    context_maps: Callable[[tuple[int, ...]], tuple[dict[int, int], dict[int, int]]],
    phrase_tokens: Callable[[int], tuple[int, ...]],
    finalize_ranked: Callable[[list[str], dict[tuple[int, ...], float], int, float], list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    if max_words < 1:
        return []

    beam = [BeamState(token_seq=token_seq, prior_score=prior_score) for token_seq, prior_score in initial_beam]
    if seed_with_empty:
        beam.append(BeamState(token_seq=(), prior_score=0.0))

    candidates: dict[tuple[int, ...], float] = {}
    for state in beam:
        if state.token_seq:
            _remember_candidate(candidates, state.token_seq, state.prior_score)

    depth = 0
    while beam and depth < max_words:
        ranked_beam = sorted(
            beam,
            key=lambda state: beam_rank(history_tokens, state.token_seq, state.prior_score),
            reverse=True,
        )[: suggest_config.beam_width]

        next_beam: list[BeamState] = []
        for state in ranked_beam:
            if len(state.token_seq) >= max_words:
                continue

            extended_history = list(history_ids) + list(state.token_seq)
            history_key = best_history_key(extended_history)
            context_token_scores, context_phrase_scores = context_maps(history_key)
            remaining = max_words - len(state.token_seq)

            for phrase_id, phrase_score in list(context_phrase_scores.items())[: suggest_config.phrase_branch_limit]:
                seq = state.token_seq + phrase_tokens(phrase_id)[:remaining]
                num_added = len(seq) - len(state.token_seq)
                if num_added == 0:
                    continue
                divisor = 1024.0 * num_added if suggest_config.normalize_phrase_scores_by_length else 1024.0
                _remember_candidate(candidates, seq, state.prior_score + (phrase_score / divisor))

            for token_id, token_score in list(context_token_scores.items())[: suggest_config.token_branch_limit]:
                seq = state.token_seq + (token_id,)
                score = state.prior_score + (token_score / 1024.0)
                _remember_candidate(candidates, seq, score)
                if len(seq) < max_words:
                    next_beam.append(BeamState(token_seq=seq, prior_score=score))

        beam = next_beam
        depth += 1

    return finalize_ranked(history_tokens, candidates, max_words, prefer_long)
