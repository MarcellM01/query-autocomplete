from __future__ import annotations

from query_autocomplete.config import BuildConfig, SuggestConfig
from query_autocomplete.generation.beam import beam_generate
from query_autocomplete.indexing.prefix_index import PrefixIndex
from query_autocomplete.models import (
    BinaryIndexData,
    EDGE_TYPE_PHRASE,
    EDGE_TYPE_TOKEN,
    ExpansionStep,
    PrefixMatchDiagnostic,
    ScoreBreakdown,
    SuggestionDiagnostic,
)
from query_autocomplete.preprocessing.preprocess import tokenize_text
from query_autocomplete.scoring.local_scorer import LocalNgramScorer

_REDUNDANT_CONTEXT_WINDOW = 3
_FUZZY_PREFIX_PRIOR_PENALTY = 0.75


class RuntimeIndex:
    def __init__(
        self,
        data: BinaryIndexData,
        *,
        build_config: BuildConfig,
        suggest_config: SuggestConfig,
        prefix_index: PrefixIndex | None = None,
    ) -> None:
        self._data = data
        self._build_config = build_config
        self._suggest_config = suggest_config
        self._token_cache: dict[int, str] = {}
        self._prefix_index = prefix_index
        self._token_to_id = {self._decode_token(token_id): token_id for token_id in range(len(self._data.vocab_entries))}
        self._scorer = LocalNgramScorer.from_payload(data.scorer_payload) if data.scorer_payload else LocalNgramScorer()
        self._phrase_by_token_ids = {phrase.token_ids: phrase for phrase in self._data.phrase_entries}

    def suggest(
        self,
        text: str,
        *,
        topk: int,
        max_words: int,
        prefer_long: float,
        suggest_config: SuggestConfig,
    ) -> list[str]:
        ranked = self.suggest_scored(
            text,
            topk=topk,
            max_words=max_words,
            prefer_long=prefer_long,
            suggest_config=suggest_config,
        )
        return [text for text, _score in ranked[: max(1, topk)]]

    def suggest_scored(
        self,
        text: str,
        *,
        topk: int,
        max_words: int,
        prefer_long: float,
        suggest_config: SuggestConfig,
    ) -> list[tuple[str, float]]:
        ranked = self.inspect(
            text,
            topk=topk,
            max_words=max_words,
            prefer_long=prefer_long,
            suggest_config=suggest_config,
        )
        return [(item.text, item.score) for item in ranked]

    def inspect(
        self,
        text: str,
        *,
        topk: int,
        max_words: int,
        prefer_long: float,
        suggest_config: SuggestConfig,
    ) -> list[SuggestionDiagnostic]:
        trailing_ws = bool(text) and text[-1].isspace()
        tokens = tokenize_text(text, config=self._build_config.normalization)
        if trailing_ws:
            history_tokens = tokens
            fragment = ""
        else:
            history_tokens = tokens[:-1] if tokens else []
            fragment = tokens[-1] if tokens else ""
        history_ids = self._tokens_to_ids(history_tokens)
        if self._should_suppress_for_low_context_support(history_ids, suggest_config=suggest_config):
            return []
        suggestions = (
            self._suggest_partial(
                history_tokens,
                history_ids,
                fragment,
                max_words=max_words,
                prefer_long=prefer_long,
                suggest_config=suggest_config,
            )
            if fragment
            else self._suggest_continuation(
                history_tokens,
                history_ids,
                max_words=max_words,
                prefer_long=prefer_long,
                suggest_config=suggest_config,
            )
        )
        if suggest_config.collapse_prefix_ladders:
            suggestions = self._collapse_prefix_ladders(suggestions, suggest_config=suggest_config)
        return suggestions[: max(1, topk)]

    def _suggest_partial(
        self,
        history_tokens: list[str],
        history_ids: list[int | None],
        fragment: str,
        *,
        max_words: int,
        prefer_long: float,
        suggest_config: SuggestConfig,
    ) -> list[SuggestionDiagnostic]:
        block_matches = self._lookup_prefix_blocks(fragment, suggest_config=suggest_config)
        if not block_matches:
            return []
        history_key = self._best_history_key(history_ids, suggest_config=suggest_config)
        context_token_scores, _context_phrase_scores = self._context_maps(history_key)
        merged_beam: dict[tuple[int, ...], tuple[float, PrefixMatchDiagnostic]] = {}
        for block_id, edit_distance, matched_prefix in block_matches:
            penalty = _FUZZY_PREFIX_PRIOR_PENALTY * edit_distance
            prefix_match = PrefixMatchDiagnostic(
                fragment=fragment,
                matched=matched_prefix,
                edit_distance=edit_distance,
                fuzzy=edit_distance > 0,
            )
            for posting in self._data.token_postings[block_id]:
                prior = (
                    (self._score_value(posting.score_id) + context_token_scores.get(posting.token_id, 0)) / 1024.0
                ) - penalty
                token_seq = (posting.token_id,)
                existing = merged_beam.get(token_seq)
                if existing is None or prior > existing[0]:
                    merged_beam[token_seq] = (prior, prefix_match)
        beam = [(token_seq, prior) for token_seq, (prior, _prefix_match) in merged_beam.items()]
        prefix_matches = {token_seq: prefix_match for token_seq, (_prior, prefix_match) in merged_beam.items()}
        return beam_generate(
            history_tokens,
            history_ids,
            initial_beam=beam,
            max_words=max_words,
            prefer_long=prefer_long,
            suggest_config=suggest_config,
            seed_with_empty=False,
            beam_rank=lambda hist, seq, prior: self._beam_rank(hist, seq, prior, suggest_config=suggest_config),
            best_history_key=lambda ids: self._best_history_key(ids, suggest_config=suggest_config),
            context_maps=self._context_maps,
            phrase_tokens=self._phrase_tokens,
            finalize_ranked=lambda hist, cands, mw, pl: self._finalize_ranked_from_beam(
                hist,
                cands,
                mw,
                pl,
                suggest_config=suggest_config,
                prefix_matches=prefix_matches,
            ),
        )

    def _suggest_continuation(
        self,
        history_tokens: list[str],
        history_ids: list[int | None],
        *,
        max_words: int,
        prefer_long: float,
        suggest_config: SuggestConfig,
    ) -> list[SuggestionDiagnostic]:
        return beam_generate(
            history_tokens,
            history_ids,
            initial_beam=[],
            max_words=max_words,
            prefer_long=prefer_long,
            suggest_config=suggest_config,
            seed_with_empty=True,
            beam_rank=lambda hist, seq, prior: self._beam_rank(hist, seq, prior, suggest_config=suggest_config),
            best_history_key=lambda ids: self._best_history_key(ids, suggest_config=suggest_config),
            context_maps=self._context_maps,
            phrase_tokens=self._phrase_tokens,
            finalize_ranked=lambda hist, cands, mw, pl: self._finalize_ranked_from_beam(
                hist,
                cands,
                mw,
                pl,
                suggest_config=suggest_config,
                prefix_matches={},
            ),
        )

    def _finalize_ranked(
        self,
        history_tokens: list[str],
        candidates: dict[tuple[int, ...], float],
        *,
        max_words: int,
        prefer_long: float,
        suggest_config: SuggestConfig,
        prefix_matches: dict[tuple[int, ...], PrefixMatchDiagnostic] | None = None,
    ) -> list[SuggestionDiagnostic]:
        seen: dict[str, SuggestionDiagnostic] = {}
        prefix_matches = prefix_matches or {}
        context_ratio = self._context_support_ratio(self._tokens_to_ids(history_tokens))
        context_penalty = suggest_config.context_support_penalty_weight * (1.0 - context_ratio)
        for token_seq, prior_score in candidates.items():
            suggestion = self._compose_suggestion(history_tokens, token_seq)
            if not suggestion:
                continue
            if suggest_config.suppress_redundant_continuations and self._is_redundant_continuation(
                history_tokens, token_seq
            ):
                continue
            scorer_score = self._scorer.score(suggestion)
            noise_penalty = suggest_config.noise_penalty_weight * self._suggestion_noise_penalty(suggestion)
            base_score = scorer_score + (suggest_config.prior_weight * prior_score) - noise_penalty - context_penalty
            score = self._length_adjust(base_score, len(token_seq), max_words=max_words, prefer_long=prefer_long)
            length_adjustment = score - base_score
            diagnostic = SuggestionDiagnostic(
                text=suggestion,
                score=score,
                breakdown=ScoreBreakdown(
                    final_score=score,
                    prior_score=prior_score,
                    scorer_score=scorer_score,
                    noise_penalty=noise_penalty,
                    context_support_ratio=context_ratio,
                    context_support_penalty=context_penalty,
                    length_adjustment=length_adjustment,
                ),
                diversity_group_key=_diversity_group_key(suggestion),
                expansion_trace=self._expansion_trace(token_seq),
                prefix_match=self._prefix_match_for_token_seq(token_seq, prefix_matches),
            )
            existing = seen.get(suggestion)
            if existing is None or score > existing.score:
                seen[suggestion] = diagnostic
        return sorted(seen.values(), key=lambda item: (-item.score, item.text))

    def _finalize_ranked_from_beam(
        self,
        history_tokens: list[str],
        candidates: dict[tuple[int, ...], float],
        max_words: int,
        prefer_long: float,
        *,
        suggest_config: SuggestConfig,
        prefix_matches: dict[tuple[int, ...], PrefixMatchDiagnostic],
    ) -> list[SuggestionDiagnostic]:
        return self._finalize_ranked(
            history_tokens,
            candidates,
            max_words=max_words,
            prefer_long=prefer_long,
            suggest_config=suggest_config,
            prefix_matches=prefix_matches,
        )

    def _compose_suggestion(self, history_tokens: list[str], token_seq: tuple[int, ...]) -> str:
        return " ".join([*history_tokens, *(self._decode_token(token_id) for token_id in token_seq)]).strip()

    def _beam_rank(self, history_tokens: list[str], token_seq: tuple[int, ...], prior_score: float, *, suggest_config: SuggestConfig) -> float:
        return self._rescore_suggestion(self._compose_suggestion(history_tokens, token_seq), prior_score, suggest_config=suggest_config)

    def _phrase_tokens(self, phrase_id: int) -> tuple[int, ...]:
        return self._data.phrase_entries[phrase_id].token_ids

    def _rescore_suggestion(self, suggestion: str, prior_score: float, *, suggest_config: SuggestConfig) -> float:
        penalty = suggest_config.noise_penalty_weight * self._suggestion_noise_penalty(suggestion)
        return self._scorer.score(suggestion) + (suggest_config.prior_weight * prior_score) - penalty

    def _context_maps(self, history_key: tuple[int, ...]) -> tuple[dict[int, int], dict[int, int]]:
        edges = self._data.context_edges.get(history_key, [])
        token_scores: dict[int, int] = {}
        phrase_scores: dict[int, int] = {}
        for edge in edges:
            score = self._score_value(edge.score_id)
            if edge.target_type == EDGE_TYPE_TOKEN:
                token_scores[edge.target_id] = score
            elif edge.target_type == EDGE_TYPE_PHRASE:
                phrase_scores[edge.target_id] = score
        return token_scores, phrase_scores

    def _best_history_key(
        self,
        history_ids: list[int | None] | list[int],
        *,
        suggest_config: SuggestConfig,
    ) -> tuple[int, ...]:
        values: list[int] = []
        for token_id in reversed(history_ids):
            if token_id is None:
                break
            values.append(token_id)
        values.reverse()
        for width in range(min(self._build_config.max_context_tokens, len(values)), 0, -1):
            key = tuple(values[-width:])
            if key in self._data.context_edges:
                return key
        if suggest_config.unknown_context_strategy == "skip":
            known_values = [token_id for token_id in history_ids if token_id is not None]
            for width in range(min(self._build_config.max_context_tokens, len(known_values)), 0, -1):
                key = tuple(known_values[-width:])
                if key in self._data.context_edges:
                    return key
        return ()

    def _tokens_to_ids(self, tokens: list[str]) -> list[int | None]:
        return [self._token_to_id.get(token) for token in tokens]

    def _lookup_prefix_blocks(self, fragment: str, *, suggest_config: SuggestConfig) -> list[tuple[int, int, str]]:
        if self._prefix_index is not None:
            return self._prefix_index.lookup_prefix_blocks(
                fragment,
                fuzzy_prefix=suggest_config.fuzzy_prefix,
                max_edit_distance=suggest_config.max_edit_distance,
            )
        block_id = self._data.prefix_to_block.get(fragment)
        return [] if block_id is None else [(block_id, 0, fragment)]

    def _decode_token(self, token_id: int) -> str:
        cached = self._token_cache.get(token_id)
        if cached is not None:
            return cached
        entry = self._data.vocab_entries[token_id]
        start = entry.offset
        end = start + entry.length
        decoded = self._data.vocab_strings[start:end].decode("utf-8")
        self._token_cache[token_id] = decoded
        return decoded

    def _score_value(self, score_id: int) -> int:
        return int(self._data.serve_scores[score_id])

    def _should_suppress_for_low_context_support(
        self,
        history_ids: list[int | None],
        *,
        suggest_config: SuggestConfig,
    ) -> bool:
        ratio_threshold = float(suggest_config.min_context_support_ratio)
        if ratio_threshold <= 0:
            return False
        ratio = self._context_support_ratio(history_ids)
        return ratio < ratio_threshold

    def _context_support_ratio(self, history_ids: list[int | None]) -> float:
        known_ids = [token_id for token_id in history_ids if token_id is not None]
        if len(known_ids) < 2:
            return 1.0
        supported = 0
        for left, right in zip(known_ids, known_ids[1:]):
            token_scores, _phrase_scores = self._context_maps((left,))
            if right in token_scores:
                supported += 1
        return supported / max(1, len(known_ids) - 1)

    def _is_redundant_continuation(self, history_tokens: list[str], token_seq: tuple[int, ...]) -> bool:
        if not token_seq:
            return False
        first_token = self._decode_token(token_seq[0])
        normalized_first = _normalize_for_redundancy(first_token)
        if not normalized_first:
            return True
        recent = [
            normalized
            for token in history_tokens[-_REDUNDANT_CONTEXT_WINDOW:]
            if (normalized := _normalize_for_redundancy(token))
        ]
        if normalized_first in recent:
            return True
        for width in range(2, min(len(recent), _REDUNDANT_CONTEXT_WINDOW) + 1):
            if normalized_first == "".join(recent[-width:]):
                return True
        return False

    def _suggestion_noise_penalty(self, suggestion: str) -> float:
        tokens = suggestion.split()
        if not tokens:
            return 0.0
        return max(_token_noise_penalty(token) for token in tokens)

    def _expansion_trace(self, token_seq: tuple[int, ...]) -> tuple[ExpansionStep, ...]:
        steps: list[ExpansionStep] = []
        index = 0
        while index < len(token_seq):
            phrase = None
            for end in range(len(token_seq), index + 1, -1):
                phrase = self._phrase_by_token_ids.get(token_seq[index:end])
                if phrase is not None:
                    break
            if phrase is not None:
                steps.append(
                    ExpansionStep(
                        kind="phrase",
                        text=" ".join(self._decode_token(token_id) for token_id in phrase.token_ids),
                        token_ids=phrase.token_ids,
                    )
                )
                index += len(phrase.token_ids)
                continue
            token_id = token_seq[index]
            steps.append(ExpansionStep(kind="token", text=self._decode_token(token_id), token_ids=(token_id,)))
            index += 1
        return tuple(steps)

    def _collapse_prefix_ladders(
        self,
        diagnostics: list[SuggestionDiagnostic],
        *,
        suggest_config: SuggestConfig,
    ) -> list[SuggestionDiagnostic]:
        if suggest_config.collapse_prefix_ladder_strategy != "best":
            diagnostics = sorted(
                diagnostics,
                key=lambda item: (
                    -len(tuple(tokenize_text(item.text, config=self._build_config.normalization)))
                    if suggest_config.collapse_prefix_ladder_strategy == "prefer_longest"
                    else len(tuple(tokenize_text(item.text, config=self._build_config.normalization))),
                    -item.score,
                    item.text,
                ),
            )
        kept: list[SuggestionDiagnostic] = []
        kept_tokens: list[tuple[str, ...]] = []
        for item in diagnostics:
            tokens = tuple(tokenize_text(item.text, config=self._build_config.normalization))
            if any(_is_prefix_ladder(tokens, existing) for existing in kept_tokens):
                continue
            kept.append(item)
            kept_tokens.append(tokens)
        return sorted(kept, key=lambda item: (-item.score, item.text))

    @staticmethod
    def _prefix_match_for_token_seq(
        token_seq: tuple[int, ...],
        prefix_matches: dict[tuple[int, ...], PrefixMatchDiagnostic],
    ) -> PrefixMatchDiagnostic | None:
        for width in range(len(token_seq), 0, -1):
            match = prefix_matches.get(token_seq[:width])
            if match is not None:
                return match
        return None

    @staticmethod
    def _length_adjust(score: float, words: int, *, max_words: int, prefer_long: float) -> float:
        if max_words <= 1:
            return score
        bias = (prefer_long - 0.5) * 0.35
        return score + (bias * min(words, max_words))


def _normalize_for_redundancy(token: str) -> str:
    return "".join(char for char in token.lower() if char.isalnum())


def _diversity_group_key(suggestion: str) -> str:
    return " ".join(_normalize_for_redundancy(token) for token in suggestion.split() if token)


def _is_prefix_ladder(candidate: tuple[str, ...], existing: tuple[str, ...]) -> bool:
    if not candidate or not existing or candidate == existing:
        return False
    shorter = min(len(candidate), len(existing))
    return candidate[:shorter] == existing[:shorter]


def _token_noise_penalty(token: str) -> float:
    normalized = token.lower()
    penalty = 0.0
    if "/" in normalized or "." in normalized:
        penalty += 10.0
    if any(char.isdigit() for char in normalized):
        penalty += 8.0
    return penalty
