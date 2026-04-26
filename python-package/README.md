# query-autocomplete

[![Release to PyPI](https://github.com/MarcellM01/query-autocomplete/actions/workflows/release.yml/badge.svg)](https://github.com/MarcellM01/query-autocomplete/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/query-autocomplete.svg)](https://pypi.org/project/query-autocomplete/)
[![Python](https://img.shields.io/pypi/pyversions/query-autocomplete.svg)](https://pypi.org/project/query-autocomplete/)
[![License](https://img.shields.io/pypi/l/query-autocomplete.svg)](https://github.com/MarcellM01/query-autocomplete/blob/main/LICENSE)

Local, typo-tolerant autocomplete without Elasticsearch.

Turn your own text into fast local suggestions with a compact prefix index, fuzzy prefix recovery, and a local Kneser-Ney scorer.

The easiest way to understand it is:

1. start with one text string in memory
2. move to the document model when you want stable document IDs
3. move to a persisted document store when your data needs to live in a database

## Install

```bash
pip install query-autocomplete
```

PDF and DOCX readers are included in the base install. Optional chunking support is available for pysbd sentence segmentation:

```bash
pip install "query-autocomplete[chunking]"
```

## Basic Usage

Start with one text object and get suggestions back.

The text can be short or very long. A `Document` can be a phrase, a page, a transcript, or something closer to book length. The tiny examples here are just for readability.

```python
from query_autocomplete import Autocomplete, Document

index = Autocomplete.create([
    Document(text="how to build a deck"),
])

print(index.suggest("how to bui", topk=5))
print(index.suggest("how to biuld", topk=5))
```

That is the core experience: give it text, create an in-memory autocomplete, ask for suggestions.

## Slightly Bigger In-Memory Example

Once you want better results, just add more text or bigger documents.

```python
from query_autocomplete import Autocomplete, Document

index = Autocomplete.create([
    Document(text="how to build a deck"),
    Document(text="how to build a desk"),
    Document(text="how to build with python"),
])

print(index.suggest("how to bui", topk=5))
print(index.suggest("how to build ", topk=5))
```

This is still the simplest mode and the best place to begin.

## The Document Model

The real unit in the library is a `Document`.

```python
from query_autocomplete import Document

doc = Document(
    text="how to build with python",
    doc_id="doc-123",
    metadata={"source": "docs"},
)
```

Fields:

- `text`
  The raw text used for learning suggestions.

- `doc_id`
  Optional stable identifier for the document.

- `metadata`
  Optional JSON-like metadata kept on in-memory document objects.

For the basic in-memory flow, you usually do not need `doc_id`.

For persisted mutable stores, `doc_id` becomes important because it is the public document identity used for document management.

`Document.text` does not need to be short. It can be a single query-like phrase, a paragraph, a full article, a long transcript, or very large source text. The system is designed to adapt to mixed short and long documents in the same store.

One document can contain multiple lines. Internally, the library may split those lines for training.

## Quality Profiles

The default profile is `balanced`. It turns on conservative production-quality behavior such as context-aware scoring, typo-tolerant prefix lookup, and prefix-ladder collapse.

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

Available profiles:

- `balanced`
  The default. A conservative mix of fuzzy recall, quality filtering, and clean top results.

- `precision`
  Stricter phrase mining and stronger runtime penalties for cleaner top results.

- `recall`
  Keeps more candidates and disables prefix-ladder collapse by default. Fuzzy prefix lookup remains enabled.

- `code_or_logs`
  Keeps structured tokens and code/log-like continuations more readily.

- `natural_language`
  Uses stricter phrase and diversity behavior for prose-like document collections.

Explicit `BuildConfig` and `SuggestConfig` objects override profile defaults.

Use `inspect()` when debugging ranking. For partial-token queries, diagnostics include `prefix_match`, which reports the typed fragment, the matched indexed prefix, edit distance, and whether fuzzy recovery was used.

## Inspecting Rankings

Use `inspect(...)` when you want to understand why suggestions ranked the way they did.

```python
diagnostics = index.inspect("how to bui", topk=3)

for item in diagnostics:
    print(item.text, item.score)
    print(item.breakdown)
    print(item.expansion_trace)
```

Each diagnostic includes:

- final score
- prior score from prefix/context evidence
- local scorer score
- structural noise penalty
- context support ratio and penalty
- length adjustment
- diversity group key
- token or phrase expansion trace

`suggest(...)` still returns plain `list[str]`; diagnostics are only returned by `inspect(...)`.

## Persistence Helpers

If you want to keep a compiled autocomplete index around and load it later, you can save it as an artifact. This is a persistence helper, not the main `Autocomplete` mental model.

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

You can also create and save in one step:

```python
from query_autocomplete import Autocomplete, Document

Autocomplete.create(
    [
        Document(text="how to build a deck"),
        Document(text="how to build a desk"),
    ],
).save("my-index")
```

Path rules:

- `index.save()`
  Auto-creates a managed folder under `.query_autocomplete_artifacts/`

- `index.save("docs-v1")`
  Saves to `.query_autocomplete_artifacts/docs-v1/`

- `index.save("artifacts/docs-v1")`
  Saves to that explicit relative path

- `Autocomplete.load("docs-v1")`
  Loads from the managed artifact folder

This is persistence for a compiled serving artifact, not a mutable document database.

## Database Model

When your document collection needs to change over time, move to `AdaptiveStore`.

This is the database-backed model:

- one SQLite database is one document collection
- documents can be added and deleted over time
- the serving index is rebuilt from stored source documents
- `doc_id` is the public identity for document management

## SQL-Compatible Database

For a proper persisted mutable document collection, use the SQL-compatible store.

```python
from query_autocomplete import AdaptiveStore, Document

store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")

store.add_documents([
    Document(text="how to build a deck", doc_id="deck"),
    Document(text="how to build with python", doc_id="python"),
])

print(store.suggest("how to bui", topk=5))
```

Supported store URLs today:

- `sqlite:///adaptive.sqlite3`
- `sqlite:////absolute/path/adaptive.sqlite3`
- a plain path like `"./adaptive.sqlite3"`

Each adaptive SQLite database owns one document collection. Name the database file however you want; the documents and current serving index live inside that file.

Adaptive SQL persistence is SQL-first:

- source documents are stored in SQLite
- the compiled serving index cache is also stored in SQLite
- normal adaptive usage does not write `.query_autocomplete_artifacts`

## Working With Mutable Stores

### Ingest documents

```python
store.add_documents([
    Document(text="how to build a deck", doc_id="deck"),
    Document(text="how to build a desk", doc_id="desk"),
])
```

Rules for adaptive mutable stores:

- `doc_id` is optional on input and auto-generated when missing
- `doc_id` must be unique within the database
- document content must also be unique within the database

So these are both rejected inside one database:

- same `doc_id` with different content
- same content with a different `doc_id`

Ingesting documents automatically invalidates the serving cache, which is rebuilt on demand the next time you query.

### Delete a document

In adaptive stores, `doc_id` is the public document identity.

```python
store.remove_document("deck")
```

### List documents

```python
print(store.list_documents())
```

### Open an existing store

```python
store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")
```

### Clear a store

```python
store.clear()
```

`store.delete()` is kept as a backwards-compatible alias for `clear()`. It clears the adaptive database tables but does not remove the SQLite file.

### Migrate between SQL stores

```python
store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")
copied = store.migrate("sqlite:///adaptive-copy.sqlite3")
```

### Reuse a custom serving profile

```python
from query_autocomplete.config import SuggestConfig

autocomplete = store.with_suggest_config(SuggestConfig(default_top_k=3))
print(autocomplete.suggest("how to bui"))
```

`AdaptiveAutocomplete` also supports `inspect(...)` with the same diagnostics as the in-memory engine:

```python
for item in autocomplete.inspect("how to bui", topk=3):
    print(item.text, item.breakdown.final_score)
```

## Upgrade Path

You can export a live in-memory autocomplete into an adaptive store:

```python
from query_autocomplete import AdaptiveStore, Autocomplete, Document

engine = Autocomplete.create([
    Document(text="how to build a deck"),
])

store = AdaptiveStore.import_autocomplete(
    "sqlite:///adaptive.sqlite3",
    engine=engine,
)
```

You can also export the source documents directly:

```python
store = AdaptiveStore.open("sqlite:///adaptive.sqlite3")
store.add_documents(engine.export_documents())
```

An autocomplete loaded from `Autocomplete.load(...)` cannot be imported into an adaptive store, because artifact files are for serving and do not retain the full source-document provenance needed for mutable retraining.

## Config

You usually do not need to touch config first, but when you do:

- `BuildConfig`
  Controls index construction and compilation behavior for `AdaptiveStore`

- `SuggestConfig`
  Controls serving behavior for `store.with_suggest_config(...)`

Example:

```python
from query_autocomplete import Autocomplete, Document
from query_autocomplete.config import BuildConfig, NormalizationConfig, SuggestConfig

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
    normalization=NormalizationConfig(
        lowercase=True,
        unicode_nfkc=True,
        strip_accents=False,
        strip_punctuation=True,
    ),
)

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

index = Autocomplete.create(
    [Document(text="how to build a deck")],
    build_config=build_config,
    suggest_config=suggest_config,
)
```

Most useful knobs:

- `BuildConfig.max_generated_words`
- `BuildConfig.max_context_tokens`
  Defaults to `3`; values up to `6` are supported. Higher values are rejected because the binary context graph stores at most six-token history keys.
- `BuildConfig.phrase_min_count`
- `BuildConfig.phrase_min_doc_freq`
- `BuildConfig.phrase_min_pmi`
- `SuggestConfig.default_top_k`
- `SuggestConfig.max_suggestion_words`
- `SuggestConfig.default_length_bias`
- `SuggestConfig.context_support_penalty_weight`
- `SuggestConfig.collapse_prefix_ladders`
- `SuggestConfig.collapse_prefix_ladder_strategy`
- `SuggestConfig.unknown_context_strategy`
- `SuggestConfig.normalize_phrase_scores_by_length`
- `SuggestConfig.fuzzy_prefix`
  Defaults to `"auto"`: exact prefix lookup is tried first, then bounded fuzzy lookup recovers common one-edit typos on non-trivial fragments.
- `SuggestConfig.max_edit_distance`
  Defaults to `2`; serving may use a lower effective distance for short fragments to avoid noisy autocomplete matches.

Phrase quality options are build-time settings. Changing them requires rebuilding the index or adaptive serving artifact.

Runtime quality options are serving-time settings. You can override them per call:

```python
results = index.suggest(
    "how to build ",
    collapse_prefix_ladders=False,
)
```

`collapse_prefix_ladders` removes near-duplicate suggestions where one result is just a longer continuation of another. For example, instead of returning all of `how to build`, `how to build a`, and `how to build a deck`, the default keeps one representative according to `collapse_prefix_ladder_strategy`.

Candidate fluency is scored locally with an interpolated Kneser-Ney bigram model built from the indexed corpus. This keeps serving lightweight while giving better contextual preferences than simple add-k smoothing.

Rerankers are request-time behavior:

```python
results = index.suggest("how to build ", reranker=my_reranker)
diagnostics = index.inspect("how to build ", reranker=my_reranker)
```

If a request asks for longer continuations than the index was built for, the library emits a warning. For example, an index built with `max_generated_words=4` warns when called with `suggest(..., max_words=5)`.

The same warning behavior applies when serving asks for artifact detail that was not stored at build time: a partial query fragment longer than `BuildConfig.max_indexed_prefix_chars`, or `SuggestConfig.token_branch_limit` / `phrase_branch_limit` values larger than `BuildConfig.top_next_tokens` / `top_next_phrases`.

## Repository Note

- The published package is built from `python-package/`
- The importable library source lives in `core/src/query_autocomplete/`

## Third-Party Licensing

- This package is MIT-licensed.
- It depends on `marisa-trie`, whose current published licensing is `MIT AND (BSD-2-Clause OR LGPL-2.1-or-later)`.
- See [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for a short note and links to upstream metadata.
