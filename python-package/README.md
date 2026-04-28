# query-autocomplete

[![Release to PyPI](https://github.com/MarcellM01/query-autocomplete/actions/workflows/release.yml/badge.svg)](https://github.com/MarcellM01/query-autocomplete/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/query-autocomplete.svg)](https://pypi.org/project/query-autocomplete/)
[![Python](https://img.shields.io/pypi/pyversions/query-autocomplete.svg)](https://pypi.org/project/query-autocomplete/)
[![License](https://img.shields.io/pypi/l/query-autocomplete.svg)](https://github.com/MarcellM01/query-autocomplete/blob/main/LICENSE)

Local, typo-tolerant autocomplete for Python apps.

`query-autocomplete` turns text, PDFs, and DOCX files into fast local suggestions with a compact prefix index, fuzzy prefix recovery, and a local scorer. It is useful when you want autocomplete without running Elasticsearch, Meilisearch, Algolia, Typesense, or another search service.

Full documentation: https://query-autocomplete.readthedocs.io/en/latest/

## Install

```bash
pip install query-autocomplete
```

Optional sentence chunking support:

```bash
pip install "query-autocomplete[chunking]"
```

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

That is the core model: build an index from text, then ask for suggestions.

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

## Use It In An App

Build or load the autocomplete once when your app starts. Do not rebuild it inside every request handler.

```python
from fastapi import FastAPI
from query_autocomplete import Autocomplete

app = FastAPI()
index = Autocomplete.load("my-index")

# Warm before serving traffic so the first user query is not the loader.
index.warm()

@app.get("/autocomplete")
def autocomplete(q: str):
    return {"suggestions": index.suggest(q, topk=5)}
```

Cold starts are normal for local indexes: a new process has to load the compiled index into memory once. After that, suggestions are served from the in-process engine.

## Save A Serving Index

If your source text changes rarely, build the index ahead of time and save it as a compiled artifact.

```python
from query_autocomplete import Autocomplete, Document

index = Autocomplete.create([
    Document(text="wireless mechanical keyboard"),
    Document(text="wireless mouse for laptop"),
    Document(text="usb c docking station"),
    Document(text="noise cancelling headphones"),
])

index.save("products-v1")
```

Then load it in your app:

```python
from query_autocomplete import Autocomplete

index = Autocomplete.load("products-v1")
print(index.suggest("wireless m", topk=5))
```

Use this path when you want the simplest serving setup: compile once, load many times.

## Use SQLite When Documents Change

If your document collection needs to be updated over time, use `AdaptiveStore`. It stores source documents and the current compiled serving index in SQLite.

```python
from query_autocomplete import AdaptiveStore, Document

store = AdaptiveStore.open("sqlite:///autocomplete.sqlite3")

store.add_documents([
    Document(text="how to build a deck", doc_id="deck"),
    Document(text="how to build with python", doc_id="python"),
])

# Warm before serving traffic so the first user query is not the builder.
store.warm()

print(store.suggest("how to bui", topk=5))
```

`AdaptiveStore` rebuilds the serving index when documents change. For production-style apps, call `store.warm()` during startup or after ingestion so the first real user request does not pay that cost.

Supported SQLite paths:

- `sqlite:///autocomplete.sqlite3`
- `sqlite:////absolute/path/autocomplete.sqlite3`
- a plain path like `"./autocomplete.sqlite3"`

Serving a SQLite-backed autocomplete from FastAPI:

```python
from fastapi import FastAPI
from query_autocomplete import AdaptiveStore

app = FastAPI()
store = AdaptiveStore.open("sqlite:///autocomplete.sqlite3")

@app.on_event("startup")
def startup():
    store.warm()

@app.get("/autocomplete")
def autocomplete(q: str):
    return {"suggestions": store.suggest(q, topk=5)}
```

## Which API Should I Use?

Use `Autocomplete.create(...)` when your documents are already in memory and you want the fastest way to start.

Use `Autocomplete.save(...)` and `Autocomplete.load(...)` when you want a compiled serving artifact that can be loaded at app startup.

Use `AdaptiveStore` when you want a SQLite-backed document collection that can add or remove documents and rebuild its serving index.

## What It Is Good For

- search box suggestions
- command palettes
- docs and help-center autocomplete
- local autocomplete inside Python apps
- prototypes that should not need a search server

It is probably not the right tool when you need distributed search, complex faceting, hosted multi-tenant search infrastructure, or semantic/vector search as the primary retrieval model.

## Learn More

The full guide covers configuration, quality profiles, diagnostics, file readers, adaptive stores, and persistence:

https://query-autocomplete.readthedocs.io/en/latest/

## Repository Note

- The published package is built from `python-package/`
- The importable library source lives in `core/src/query_autocomplete/`

## Third-Party Licensing

- This package is MIT-licensed.
- It depends on `marisa-trie`, whose current published licensing is `MIT AND (BSD-2-Clause OR LGPL-2.1-or-later)`.
- See [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for a short note and links to upstream metadata.
