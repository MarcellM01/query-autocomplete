# query-autocomplete User Guide

`query-autocomplete` builds fast local autocomplete suggestions from your own text, PDFs, and DOCX files. Use it for search boxes, command palettes, document title completion, and typo-tolerant prefix search without running Elasticsearch, Meilisearch, Algolia, Typesense, or another search server.

## Install

```bash
pip install query-autocomplete
```

PDF and DOCX readers are included in the base install. Optional sentence chunking support is available with:

```bash
pip install "query-autocomplete[chunking]"
```

## Choose a Path

The library has three common ways to use it:

- `Autocomplete`: build an in-memory autocomplete from documents
- saved artifacts: save a compiled serving index and load it later
- `AdaptiveStore`: keep source documents in SQLite and update them over time

Start with `Autocomplete`. Move to saved artifacts when you want to build once and serve many times. Move to `AdaptiveStore` when your documents need to be added, removed, listed, or persisted as source data.

Use `Autocomplete` when:

- your source text can be loaded at startup
- the document set rarely changes
- you want the simplest possible integration

Use saved artifacts when:

- you build an index once
- you serve it many times
- you do not need to mutate the source documents

Use `AdaptiveStore` when:

- documents are added or removed over time
- you need durable source documents
- you want SQLite-backed persistence

## Cold Starts

Build or load the autocomplete once when your app starts. Do not rebuild it inside every request handler.

Cold starts happen per process: a new process has to load or rebuild the serving index once. After that, suggestions are served from the in-process engine.

For app startup, call `warm()` after loading the index or opening the store:

```python
from query_autocomplete import Autocomplete

index = Autocomplete.load("my-index")
index.warm()
```

For mutable SQLite stores:

```python
from query_autocomplete import AdaptiveStore

store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")
store.warm()
```

`AdaptiveStore.warm()` loads the current compiled serving index if one exists, or rebuilds it from stored documents if needed. If the store has no documents yet, it is a no-op.

## Quick Start

```python
from query_autocomplete import Autocomplete, Document

index = Autocomplete.create([
    Document(text="how to build a deck"),
    Document(text="how to build a desk"),
    Document(text="how to build with python"),
])

print(index.suggest("how to bui", topk=5))
print(index.suggest("how to biuld", topk=5))
```

`suggest(...)` returns a plain `list[str]`.

You can also pass file paths directly. `.txt`, `.pdf`, and `.docx` inputs are supported in the base package:

```python
from pathlib import Path
from query_autocomplete import Autocomplete

index = Autocomplete.create([
    Path("docs/handbook.pdf"),
    Path("docs/release-notes.docx"),
    Path("docs/faq.txt"),
])

print(index.suggest("install", topk=5))
```

## Documents

The main input type is `Document`.

```python
from query_autocomplete import Document

doc = Document(
    text="how to build with python",
    doc_id="doc-123",
    metadata={"source": "docs"},
)
```

Fields:

- `text`: source text used to learn suggestions
- `doc_id`: optional stable identifier
- `metadata`: optional JSON-like metadata on in-memory documents

`Document.text` can be a phrase, paragraph, full article, transcript, or larger body of text. Use `doc_id` when you need stable document identity, especially with `AdaptiveStore`.

## In-Memory Autocomplete

Use `Autocomplete` when the source collection can be loaded in memory.

```python
from query_autocomplete import Autocomplete, Document

documents = [
    Document(text="how to build a deck"),
    Document(text="how to build a desk"),
    Document(text="how to build with python"),
]

index = Autocomplete.create(documents)

print(index.suggest("how to build ", topk=5))
```

Useful methods:

```python
index = Autocomplete.create(documents)
index.suggest("how to bui", topk=5)
index.inspect("how to bui", topk=5)
index.warm()
index.save("my-index")
loaded = Autocomplete.load("my-index")
docs = index.export_documents()
```

## Saved Artifacts

Saved artifacts are compiled serving indexes. They are useful when you build an autocomplete once and load it later.

```python
from query_autocomplete import Autocomplete, Document

index = Autocomplete.create([
    Document(text="how to build a deck"),
    Document(text="how to build a desk"),
])

index.save("my-index")

loaded = Autocomplete.load("my-index")
print(loaded.suggest("how to bui", topk=5))
```

Artifact path behavior:

- `index.save()` writes to a managed folder under `.query_autocomplete_artifacts/`
- `index.save("docs-v1")` writes to `.query_autocomplete_artifacts/docs-v1/`
- `index.save("artifacts/docs-v1")` writes to that explicit relative path
- `Autocomplete.load("docs-v1")` loads from the managed artifact folder

Artifacts are for serving. They do not act like a mutable document database.

## SQLite Adaptive Stores

Use `AdaptiveStore` when documents change over time.

```python
from query_autocomplete import AdaptiveStore, Document

store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")

store.add_documents([
    Document(text="how to build a deck", doc_id="deck"),
    Document(text="how to build with python", doc_id="python"),
])

print(store.suggest("how to bui", topk=5))
```

Each adaptive SQLite database owns one document collection. Adding documents invalidates the serving cache, which is rebuilt when needed.

Supported store paths:

```python
AdaptiveStore.open("sqlite:///adaptive.sqlite3")
AdaptiveStore.open("sqlite:////absolute/path/adaptive.sqlite3")
AdaptiveStore.open("./adaptive.sqlite3")
AdaptiveStore.open(":memory:")
```

Serving a SQLite-backed autocomplete from FastAPI:

```python
from fastapi import FastAPI
from query_autocomplete import AdaptiveStore

app = FastAPI()
store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")

@app.on_event("startup")
def startup():
    store.warm()

@app.get("/autocomplete")
def autocomplete(q: str):
    return {"suggestions": store.suggest(q, topk=5)}
```

Useful methods:

```python
store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")
store = AdaptiveStore.open_or_create("sqlite:///adaptive.sqlite3")

result = store.add_documents([
    Document(text="how to build a deck", doc_id="deck"),
])

store.suggest("how to bui", topk=5)
store.inspect("how to bui", topk=5)
store.warm()
store.list_documents()
store.remove_document("deck")
store.clear()
store.migrate("sqlite:///adaptive-copy.sqlite3")
```

`store.delete()` is available as a backwards-compatible alias for `store.clear()`.

Use `AdaptiveStore.import_autocomplete(...)` to promote an in-memory engine into a SQLite-backed store:

```python
store = AdaptiveStore.import_autocomplete(
    "sqlite:///adaptive.sqlite3",
    engine=index,
)
```

## Reusable Serving Config

Use `store.with_suggest_config(...)` when you want to reuse runtime serving settings.

```python
from query_autocomplete import SuggestConfig

autocomplete = store.with_suggest_config(SuggestConfig(default_top_k=3))
autocomplete.suggest("how to bui")
autocomplete.inspect("how to bui")
```

This returns an `AdaptiveAutocomplete` handle backed by the same store.

## Quality Profiles

Most projects should start with a quality profile before touching individual config fields.

```python
from query_autocomplete import Autocomplete, Document

index = Autocomplete.create(
    [
        Document(text="how to build a deck"),
        Document(text="how to build a desk"),
        Document(text="how to build with python"),
    ],
    quality_profile="precision",
    max_generated_words=4,
    phrase_min_count=3,
)
```

Profiles:

- `balanced`: default behavior for clean suggestions
- `precision`: stricter ranking and phrase mining
- `recall`: keeps more candidates
- `code_or_logs`: better for structured tokens, code, and logs
- `natural_language`: better for prose-like collections

Explicit `BuildConfig` and `SuggestConfig` values override profile defaults.

## Inspect Rankings

Use `inspect(...)` when you want to understand why suggestions ranked the way they did.

```python
diagnostics = index.inspect("how to bui", topk=3)

for item in diagnostics:
    print(item.text, item.score)
    print(item.breakdown)
    print(item.expansion_trace)
```

Diagnostics include score details, prefix matching information, and expansion traces. `suggest(...)` still returns plain strings.

## Custom Reranking

You can pass a reranker to `suggest(...)` or `inspect(...)`.

```python
from query_autocomplete import BaseReranker

class ReverseReranker(BaseReranker):
    def rerank(self, prefix: str, candidates: list[str]) -> list[str]:
        return list(reversed(candidates))

results = index.suggest("how to build ", reranker=ReverseReranker())
diagnostics = index.inspect("how to build ", reranker=ReverseReranker())
```

## Configuration Reference

There are three config layers:

- `BuildConfig`: build-time indexing, phrase mining, and pruning
- `SuggestConfig`: runtime ranking and generation
- `NormalizationConfig`: text normalization before indexing

### `BuildConfig`

```python
from query_autocomplete.config import BuildConfig, NormalizationConfig

build_config = BuildConfig(
    max_generated_words=4,
    max_indexed_prefix_chars=24,
    max_context_tokens=3,
    top_tokens_per_prefix=64,
    top_next_tokens=32,
    top_next_phrases=16,
    phrase_min_count=2,
    phrase_min_doc_freq=1,
    phrase_min_pmi=0.0,
    phrase_max_dominant_extension_ratio=0.95,
    phrase_boundary_generic_min_count=8,
    phrase_max_len=4,
    vocab_prune_min_total_tokens=100_000,
    vocab_prune_min_unigram_count=2,
    vocab_prune_min_segment_freq=2,
    vocab_prune_rescue_unigram=12,
    vocab_prune_line_count_to_apply_df=5_000,
    normalization=NormalizationConfig(),
)
```

Fields:

- `max_generated_words`: maximum generated continuation length stored in the index
- `max_indexed_prefix_chars`: maximum prefix length indexed for lookup
- `max_context_tokens`: number of previous tokens used for context, up to `6`
- `top_tokens_per_prefix`: number of token candidates retained per prefix
- `top_next_tokens`: number of next-token transitions retained
- `top_next_phrases`: number of phrase transitions retained
- `phrase_min_count`: minimum phrase count for phrase mining
- `phrase_min_doc_freq`: minimum document frequency for phrases
- `phrase_min_pmi`: minimum PMI score for phrases
- `phrase_max_dominant_extension_ratio`: filters phrases dominated by one extension
- `phrase_boundary_generic_min_count`: filters generic phrase boundaries
- `phrase_max_len`: maximum mined phrase length
- `vocab_prune_min_total_tokens`: corpus size threshold before vocabulary pruning activates
- `vocab_prune_min_unigram_count`: minimum unigram count when pruning
- `vocab_prune_min_segment_freq`: minimum segment frequency when pruning
- `vocab_prune_rescue_unigram`: keep words that are frequent enough even if segment frequency is low
- `vocab_prune_line_count_to_apply_df`: segment-count threshold before segment-frequency pruning applies

### `SuggestConfig`

```python
from query_autocomplete import SuggestConfig

suggest_config = SuggestConfig(
    default_top_k=10,
    default_length_bias=0.5,
    max_suggestion_words=4,
    beam_width=24,
    token_branch_limit=8,
    phrase_branch_limit=8,
    prior_weight=0.35,
    noise_penalty_weight=0.35,
    suppress_redundant_continuations=True,
    min_context_support_ratio=0.0,
    context_support_penalty_weight=0.25,
    collapse_prefix_ladders=True,
    collapse_prefix_ladder_strategy="best",
    unknown_context_strategy="skip",
    normalize_phrase_scores_by_length=False,
    fuzzy_prefix="auto",
    max_edit_distance=2,
)
```

Fields:

- `default_top_k`: default number of suggestions
- `default_length_bias`: preference for shorter or longer completions
- `max_suggestion_words`: maximum words returned at serving time
- `beam_width`: search width during generation
- `token_branch_limit`: token candidates explored per beam step
- `phrase_branch_limit`: phrase candidates explored per beam step
- `prior_weight`: weight for prefix and context evidence
- `noise_penalty_weight`: weight for structural noise penalties
- `suppress_redundant_continuations`: suppress near-duplicate continuations
- `min_context_support_ratio`: minimum context support before penalties apply
- `context_support_penalty_weight`: strength of context-support penalties
- `collapse_prefix_ladders`: collapse suggestions that are just longer versions of each other
- `collapse_prefix_ladder_strategy`: `best`, `prefer_longest`, or `prefer_shortest`
- `unknown_context_strategy`: `skip` or `strict`
- `normalize_phrase_scores_by_length`: normalize phrase scores by phrase length
- `fuzzy_prefix`: `auto`, `True`, or `False`
- `max_edit_distance`: maximum fuzzy prefix edit distance

### `NormalizationConfig`

```python
from query_autocomplete.config import NormalizationConfig

normalization = NormalizationConfig(
    lowercase=True,
    unicode_nfkc=True,
    strip_accents=False,
    strip_punctuation=True,
    split_sentences=True,
    pysbd_language=None,
)
```

Set `pysbd_language` to a language code such as `"en"` only if you installed sentence chunking support:

```bash
pip install "query-autocomplete[chunking]"
```

## Public API Reference

Most users should import from the top-level package:

```python
from query_autocomplete import (
    AdaptiveAutocomplete,
    AdaptiveStore,
    Autocomplete,
    BaseReranker,
    BuildConfig,
    DeleteResult,
    Document,
    ExpansionStep,
    HeuristicReranker,
    IngestResult,
    QualityProfile,
    ScoreBreakdown,
    SuggestConfig,
    SuggestionDiagnostic,
    apply_quality_profile,
)
```

Main objects:

- `Autocomplete`: in-memory autocomplete engine
- `AdaptiveStore`: SQLite-backed mutable document store
- `AdaptiveAutocomplete`: serving handle returned by `store.with_suggest_config(...)`
- `Document`: source text plus optional `doc_id` and `metadata`
- `BuildConfig`: build-time indexing and phrase-mining settings
- `SuggestConfig`: runtime suggestion and ranking settings
- `QualityProfile`: one of `balanced`, `precision`, `recall`, `code_or_logs`, or `natural_language`
- `IngestResult`: returned by `store.add_documents(...)`
- `DeleteResult`: returned by `store.remove_document(...)`
- `SuggestionDiagnostic`: returned by `inspect(...)`
- `ScoreBreakdown`: diagnostic score details
- `ExpansionStep`: diagnostic expansion trace item
- `BaseReranker`: base class for custom rerankers
- `HeuristicReranker`: built-in heuristic reranker
- `apply_quality_profile`: helper for applying profile defaults to configs

## More Information

See the project README for the full API walkthrough and release notes:

https://github.com/MarcellM01/query-autocomplete
