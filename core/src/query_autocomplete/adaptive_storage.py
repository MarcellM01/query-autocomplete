from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.parse import unquote, urlparse

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
    decode_context_graph_bytes,
    decode_manifest_text,
    decode_phrase_lexicon_bytes,
    decode_scorer_payload_text,
    decode_serve_scores_bytes,
    decode_token_postings_bytes,
    decode_vocab_bin_bytes,
    encode_context_graph,
    encode_manifest_text,
    encode_phrase_lexicon,
    encode_scorer_payload_text,
    encode_serve_scores,
    encode_token_postings,
    encode_vocab_bin,
)
from query_autocomplete.config import BuildConfig, NormalizationConfig, SuggestConfig
from query_autocomplete.models import BinaryIndexData, Document


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_build_config(raw: str) -> BuildConfig:
    data = json.loads(raw)
    return BuildConfig(**{**data, "normalization": NormalizationConfig(**data.get("normalization", {}))})


def _load_suggest_config(raw: str) -> SuggestConfig:
    return SuggestConfig(**json.loads(raw))


def _coerce_float(value: Any, default: float = 1.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


@dataclass(frozen=True)
class StoreRecord:
    build_config: BuildConfig
    suggest_config: SuggestConfig
    current_index_id: int | None


@dataclass(frozen=True)
class CompiledIndexRecord:
    compiled_index_id: int
    build_config: BuildConfig
    suggest_config: SuggestConfig
    manifest: dict[str, object]
    scorer_k: float
    created_at: str


@dataclass(frozen=True)
class StoredDocuments:
    documents: list[Document]
    count: int


@dataclass(frozen=True)
class DuplicateDocument:
    doc_id: str | None
    text_preview: str

    def __str__(self) -> str:
        return (
            f"WARNING: duplicate content detected — "
            f"existing_doc_id={self.doc_id!r}, preview={self.text_preview!r}"
        )


@dataclass(frozen=True)
class DocumentInsertResult:
    inserted_count: int
    duplicate_doc_ids: tuple[str, ...]
    duplicate_texts: tuple[DuplicateDocument, ...]


@dataclass(frozen=True)
class DocumentDeleteResult:
    deleted_count: int


class BaseStore(Protocol):
    def cache_key(self) -> str: ...
    def create_schema(self) -> None: ...
    def open_record(self, *, build_config: BuildConfig, suggest_config: SuggestConfig) -> StoreRecord: ...
    def load_record(self) -> StoreRecord: ...
    def reset(self, *, build_config: BuildConfig, suggest_config: SuggestConfig) -> StoreRecord: ...
    def insert_documents(self, documents: list[Document]) -> DocumentInsertResult: ...
    def delete_document(self, doc_id: str) -> DocumentDeleteResult: ...
    def fetch_all_documents(self) -> StoredDocuments: ...
    def replace_current_index(self, record: StoreRecord, *, compiled: BinaryIndexData) -> CompiledIndexRecord: ...
    def clear_current_index(self) -> None: ...
    def load_current_index_record(self) -> CompiledIndexRecord | None: ...
    def load_compiled_index_binary(self, compiled_index_id: int) -> BinaryIndexData: ...
    def insert_build_history(self, *, doc_count: int, segment_count_pre_prune: int, token_pos_pre_prune: int, segment_count_post_prune: int, token_pos_post_prune: int, vocab_size: int, pruned_type_count: int, phrase_count: int, compile_ms: int, sample_suggest_ms: int) -> None: ...
    def insert_ingest_log(self, *, attempted: int, inserted: int, ingest_ms: int) -> None: ...


class SQLiteSqlStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def cache_key(self) -> str:
        return f"sqlite:{Path(self._db_path).expanduser().resolve()}"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        if self._db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS store_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            build_config_json TEXT NOT NULL,
            suggest_config_json TEXT NOT NULL,
            current_index_id INTEGER NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL UNIQUE,
            content_md5 TEXT NOT NULL UNIQUE,
            raw_text TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS compiled_indexes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            build_config_json TEXT NOT NULL,
            suggest_config_json TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            scorer_k REAL NOT NULL DEFAULT 1.0,
            vocab_strings_blob BLOB NOT NULL,
            vocab_bin_blob BLOB NOT NULL,
            prefix_json TEXT NOT NULL,
            token_postings_blob BLOB NOT NULL,
            phrase_lexicon_blob BLOB NOT NULL,
            context_graph_blob BLOB NOT NULL,
            serve_scores_blob BLOB NOT NULL,
            scorer_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS index_build_history (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            built_at                 TEXT NOT NULL,
            doc_count                INTEGER NOT NULL,
            segment_count_pre_prune  INTEGER NOT NULL,
            token_pos_pre_prune      INTEGER NOT NULL,
            segment_count_post_prune INTEGER NOT NULL,
            token_pos_post_prune     INTEGER NOT NULL,
            vocab_size               INTEGER NOT NULL,
            pruned_type_count        INTEGER NOT NULL,
            phrase_count             INTEGER NOT NULL,
            compile_ms               INTEGER NOT NULL,
            sample_suggest_ms        INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingest_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT NOT NULL,
            attempted   INTEGER NOT NULL,
            inserted    INTEGER NOT NULL,
            ingest_ms   INTEGER NOT NULL
        );
        """
        with self._connect() as conn:
            old_schema = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'corpora'").fetchone()
            if old_schema is not None:
                raise ValueError(
                    "This SQLite adaptive store uses the old corpus-based schema. "
                    "Rebuild it as a single-collection adaptive store."
                )
            conn.executescript(schema)

    def open_record(self, *, build_config: BuildConfig, suggest_config: SuggestConfig) -> StoreRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM store_metadata WHERE id = 1").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO store_metadata (id, build_config_json, suggest_config_json, current_index_id) VALUES (1, ?, ?, NULL)",
                    (json.dumps(asdict(build_config), sort_keys=True), json.dumps(asdict(suggest_config), sort_keys=True)),
                )
        return self.load_record()

    def load_record(self) -> StoreRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM store_metadata WHERE id = 1").fetchone()
        if row is None:
            raise FileNotFoundError("Adaptive store metadata not found.")
        return StoreRecord(
            build_config=_load_build_config(str(row["build_config_json"])),
            suggest_config=_load_suggest_config(str(row["suggest_config_json"])),
            current_index_id=int(row["current_index_id"]) if row["current_index_id"] is not None else None,
        )

    def reset(self, *, build_config: BuildConfig, suggest_config: SuggestConfig) -> StoreRecord:
        with self._connect() as conn:
            conn.execute("DELETE FROM compiled_indexes")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM index_build_history")
            conn.execute("DELETE FROM ingest_log")
            conn.execute("DELETE FROM store_metadata")
            conn.execute(
                "INSERT INTO store_metadata (id, build_config_json, suggest_config_json, current_index_id) VALUES (1, ?, ?, NULL)",
                (json.dumps(asdict(build_config), sort_keys=True), json.dumps(asdict(suggest_config), sort_keys=True)),
            )
        return self.load_record()

    def insert_documents(self, documents: list[Document]) -> DocumentInsertResult:
        duplicate_doc_ids: list[str] = []
        duplicate_texts: list[DuplicateDocument] = []
        inserted = 0
        with self._connect() as conn:
            for document in documents:
                if document.doc_id is None:
                    raise ValueError("Adaptive ingestion requires Document.doc_id.")
                content_md5 = _document_md5(document.text)
                try:
                    conn.execute(
                        "INSERT INTO documents (doc_id, content_md5, raw_text, ingested_at) VALUES (?, ?, ?, ?)",
                        (document.doc_id, content_md5, document.text, _utcnow_iso()),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    existing_doc_id = conn.execute(
                        "SELECT doc_id, content_md5 FROM documents WHERE doc_id = ? LIMIT 1",
                        (document.doc_id,),
                    ).fetchone()
                    existing_content = conn.execute(
                        "SELECT doc_id, raw_text FROM documents WHERE content_md5 = ? LIMIT 1",
                        (content_md5,),
                    ).fetchone()
                    if existing_doc_id is None and existing_content is None:
                        raise
                    if existing_doc_id is not None:
                        if str(existing_doc_id["content_md5"]) != content_md5:
                            raise ValueError(
                                f"Document with doc_id {document.doc_id!r} already exists with different content."
                            )
                        duplicate_doc_ids.append(document.doc_id)
                        continue
                    if existing_content is not None:
                        duplicate_texts.append(DuplicateDocument(
                            doc_id=str(existing_content["doc_id"]),
                            text_preview=str(existing_content["raw_text"])[:100],
                        ))
        return DocumentInsertResult(inserted_count=inserted, duplicate_doc_ids=tuple(duplicate_doc_ids), duplicate_texts=tuple(duplicate_texts))

    def delete_document(self, doc_id: str) -> DocumentDeleteResult:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return DocumentDeleteResult(deleted_count=int(cursor.rowcount))

    def fetch_all_documents(self) -> StoredDocuments:
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id, raw_text FROM documents ORDER BY id").fetchall()
        return StoredDocuments(
            documents=[
                Document(
                    text=str(row["raw_text"]),
                    doc_id=str(row["doc_id"]),
                )
                for row in rows
            ],
            count=len(rows),
        )

    def replace_current_index(self, record: StoreRecord, *, compiled: BinaryIndexData) -> CompiledIndexRecord:
        scorer_payload: dict[str, Any] = dict(compiled.scorer_payload or {})
        blobs = _serialize_compiled_index_blobs(compiled)
        with self._connect() as conn:
            current_index_id = conn.execute("SELECT current_index_id FROM store_metadata WHERE id = 1").fetchone()
            if current_index_id is not None and current_index_id["current_index_id"] is not None:
                conn.execute("DELETE FROM compiled_indexes WHERE id = ?", (int(current_index_id["current_index_id"]),))
            cursor = conn.execute(
                "INSERT INTO compiled_indexes (build_config_json, suggest_config_json, manifest_json, scorer_k, vocab_strings_blob, vocab_bin_blob, prefix_json, token_postings_blob, phrase_lexicon_blob, context_graph_blob, serve_scores_blob, scorer_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    json.dumps(asdict(record.build_config), sort_keys=True),
                    json.dumps(asdict(record.suggest_config), sort_keys=True),
                    encode_manifest_text(compiled.metadata),
                    _coerce_float(scorer_payload.get("discount", 0.75)),
                    blobs[VOCAB_STRINGS_FILENAME],
                    blobs[VOCAB_BIN_FILENAME],
                    blobs[PREFIX_FILENAME],
                    blobs[TOKEN_POSTINGS_FILENAME],
                    blobs[PHRASE_LEXICON_FILENAME],
                    blobs[CONTEXT_GRAPH_FILENAME],
                    blobs[SERVE_SCORES_FILENAME],
                    blobs[SCORER_FILENAME],
                    _utcnow_iso(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to insert compiled index row: no lastrowid returned.")
            compiled_index_id = int(cursor.lastrowid)
            conn.execute("UPDATE store_metadata SET current_index_id = ? WHERE id = 1", (compiled_index_id,))
            out = conn.execute("SELECT * FROM compiled_indexes WHERE id = ?", (compiled_index_id,)).fetchone()
        if out is None:
            raise RuntimeError("Failed to persist compiled index.")
        return _compiled_index_from_mapping(dict(out))

    def clear_current_index(self) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT current_index_id FROM store_metadata WHERE id = 1").fetchone()
            if row is None or row["current_index_id"] is None:
                return
            compiled_index_id = int(row["current_index_id"])
            conn.execute("DELETE FROM compiled_indexes WHERE id = ?", (compiled_index_id,))
            conn.execute("UPDATE store_metadata SET current_index_id = NULL WHERE id = 1")

    def load_current_index_record(self) -> CompiledIndexRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ci.id, ci.build_config_json, ci.suggest_config_json, ci.manifest_json, ci.scorer_k, ci.created_at"
                " FROM store_metadata m JOIN compiled_indexes ci ON ci.id = m.current_index_id WHERE m.id = 1"
            ).fetchone()
        return _compiled_index_from_mapping(dict(row)) if row is not None else None

    def load_compiled_index_binary(self, compiled_index_id: int) -> BinaryIndexData:
        with self._connect() as conn:
            compiled_row = conn.execute("SELECT * FROM compiled_indexes WHERE id = ?", (compiled_index_id,)).fetchone()
            if compiled_row is None:
                raise FileNotFoundError(f"Compiled index not found: {compiled_index_id}")
            row = dict(compiled_row)
        return _deserialize_compiled_index_blobs(row, error_prefix="Malformed SQL compiled index")

    def insert_build_history(self, *, doc_count: int, segment_count_pre_prune: int, token_pos_pre_prune: int, segment_count_post_prune: int, token_pos_post_prune: int, vocab_size: int, pruned_type_count: int, phrase_count: int, compile_ms: int, sample_suggest_ms: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO index_build_history (built_at, doc_count, segment_count_pre_prune, token_pos_pre_prune, segment_count_post_prune, token_pos_post_prune, vocab_size, pruned_type_count, phrase_count, compile_ms, sample_suggest_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_utcnow_iso(), doc_count, segment_count_pre_prune, token_pos_pre_prune, segment_count_post_prune, token_pos_post_prune, vocab_size, pruned_type_count, phrase_count, compile_ms, sample_suggest_ms),
            )

    def insert_ingest_log(self, *, attempted: int, inserted: int, ingest_ms: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO ingest_log (logged_at, attempted, inserted, ingest_ms) VALUES (?, ?, ?, ?)",
                (_utcnow_iso(), attempted, inserted, ingest_ms),
            )


def open_store(store_url: str) -> BaseStore:
    if "://" not in store_url:
        return SQLiteSqlStore(str(Path(store_url).expanduser().resolve()))
    parsed = urlparse(store_url)
    if parsed.scheme == "sqlite":
        if parsed.path == "/:memory:":
            return SQLiteSqlStore(":memory:")
        path = unquote(f"/{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path)
        # sqlite:///name.db (three slashes) is a path relative to cwd, not "/name.db" at drive root.
        # sqlite:////abs/path (four+ slashes) yields parsed.path starting with "//" (POSIX absolute).
        if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        elif parsed.path.startswith("//"):
            path = path[1:]
        elif path.startswith("/"):
            path = path[1:]
        return SQLiteSqlStore(str(Path(path).expanduser().resolve()))
    raise NotImplementedError(f"Unsupported store URL scheme: {parsed.scheme}")


def _compiled_index_from_mapping(row: dict[str, Any]) -> CompiledIndexRecord:
    return CompiledIndexRecord(
        compiled_index_id=int(row["id"]),
        build_config=_load_build_config(str(row["build_config_json"])),
        suggest_config=_load_suggest_config(str(row["suggest_config_json"])),
        manifest=decode_manifest_text(str(row["manifest_json"])),
        scorer_k=_coerce_float(row["scorer_k"]),
        created_at=str(row["created_at"]),
    )


def _document_md5(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _serialize_compiled_index_blobs(compiled: BinaryIndexData) -> dict[str, bytes | str]:
    return {
        VOCAB_STRINGS_FILENAME: bytes(compiled.vocab_strings),
        VOCAB_BIN_FILENAME: encode_vocab_bin(compiled.vocab_entries),
        PREFIX_FILENAME: json.dumps(compiled.prefix_to_block, sort_keys=True),
        TOKEN_POSTINGS_FILENAME: encode_token_postings(compiled.token_postings),
        PHRASE_LEXICON_FILENAME: encode_phrase_lexicon(compiled.phrase_entries),
        CONTEXT_GRAPH_FILENAME: encode_context_graph(compiled.context_edges),
        SERVE_SCORES_FILENAME: encode_serve_scores(compiled.serve_scores),
        SCORER_FILENAME: encode_scorer_payload_text(dict(compiled.scorer_payload or {})),
    }


def _deserialize_compiled_index_blobs(row: dict[str, Any], *, error_prefix: str) -> BinaryIndexData:
    manifest = decode_manifest_text(str(row["manifest_json"]))
    try:
        prefix_to_block = {str(key): int(value) for key, value in json.loads(str(row["prefix_json"])).items()}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{error_prefix}: malformed prefix mapping.") from exc
    return BinaryIndexData(
        vocab_strings=bytes(row["vocab_strings_blob"]),
        vocab_entries=decode_vocab_bin_bytes(bytes(row["vocab_bin_blob"]), error_label=f"{error_prefix}: malformed vocabulary metadata."),
        serve_scores=decode_serve_scores_bytes(bytes(row["serve_scores_blob"]), error_label=f"{error_prefix}: malformed serve scores."),
        token_postings=decode_token_postings_bytes(bytes(row["token_postings_blob"]), error_label=f"{error_prefix}: malformed token postings."),
        phrase_entries=decode_phrase_lexicon_bytes(bytes(row["phrase_lexicon_blob"]), error_label=f"{error_prefix}: malformed phrase lexicon."),
        context_edges=decode_context_graph_bytes(bytes(row["context_graph_blob"]), error_label=f"{error_prefix}: malformed context graph."),
        prefix_to_block=prefix_to_block,
        scorer_payload=decode_scorer_payload_text(str(row["scorer_json"]), error_label=f"{error_prefix}: malformed scorer payload."),
        metadata=manifest,
    )
