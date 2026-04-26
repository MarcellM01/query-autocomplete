from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from query_autocomplete.models import BinaryIndexData, ContextEdge, PhraseEntry, TokenPosting, VocabEntry


MANIFEST_FILENAME = "manifest.json"
VOCAB_STRINGS_FILENAME = "vocab.strings"
VOCAB_BIN_FILENAME = "vocab.bin"
PREFIX_FILENAME = "prefix.marisa"
TOKEN_POSTINGS_FILENAME = "token_postings.bin"
PHRASE_LEXICON_FILENAME = "phrase_lexicon.bin"
CONTEXT_GRAPH_FILENAME = "context_graph.bin"
SERVE_SCORES_FILENAME = "serve_scores.bin"
SCORER_FILENAME = "scorer.json"

_VERSION = 2

_VOCAB_HEADER = struct.Struct("<4sHII")
_VOCAB_RECORD = struct.Struct("<QIIIHH")
_SCORES_HEADER = struct.Struct("<4sHI")
_TOKEN_POSTINGS_HEADER = struct.Struct("<4sHII")
_TOKEN_POSTING_RECORD = struct.Struct("<II")
_PHRASE_HEADER = struct.Struct("<4sHII")
_PHRASE_ENTRY_HEADER = struct.Struct("<HHHH")
_CONTEXT_HEADER = struct.Struct("<4sHII")
_CONTEXT_GRAPH_VERSION = 3
_CONTEXT_MAX_KEY_TOKENS = 6
_CONTEXT_NODE_RECORD_V2 = struct.Struct("<B3xIIIQ")
_CONTEXT_NODE_RECORD = struct.Struct("<B3xIIIIIIQ")
_CONTEXT_EDGE_HEADER = struct.Struct("<H2x")
_CONTEXT_EDGE_RECORD = struct.Struct("<BBHII")


def build_manifest(*, config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": _VERSION,
        "artifact_type": "autocomplete_index",
        "tokenizer": "rule_tokenizer_v2",
        "config": config,
        "score_version": "serve_v2",
        "files": {
            "vocab_strings": VOCAB_STRINGS_FILENAME,
            "vocab_bin": VOCAB_BIN_FILENAME,
            "prefix_trie": PREFIX_FILENAME,
            "token_postings": TOKEN_POSTINGS_FILENAME,
            "phrase_lexicon": PHRASE_LEXICON_FILENAME,
            "context_graph": CONTEXT_GRAPH_FILENAME,
            "serve_scores": SERVE_SCORES_FILENAME,
            "scorer": SCORER_FILENAME,
        },
        "metadata": metadata,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(encode_manifest_text(manifest), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing manifest artifact file: {path}") from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in manifest artifact file: {path}") from exc
    return _validate_manifest_object(manifest)


def encode_manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True)


def decode_manifest_text(raw: str) -> dict[str, Any]:
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid manifest: malformed JSON.") from exc
    return _validate_manifest_object(manifest)


def _validate_manifest_object(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid manifest: expected an object.")
    if manifest.get("format_version") != _VERSION:
        raise ValueError(
            f"Unsupported artifact version: {manifest.get('format_version')!r}. Expected {_VERSION}."
        )
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid manifest: missing file registry.")
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("Invalid manifest: missing config.")
    return manifest


def write_vocab_strings(path: Path, blob: bytes) -> None:
    path.write_bytes(blob)


def read_vocab_strings(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing vocabulary strings artifact file: {path}") from exc


def write_vocab_bin(path: Path, entries: list[VocabEntry]) -> None:
    path.write_bytes(encode_vocab_bin(entries))


def read_vocab_bin(path: Path) -> list[VocabEntry]:
    data = _read_required_bytes(path, "vocabulary metadata")
    return decode_vocab_bin_bytes(data, error_label=f"Malformed vocabulary metadata artifact file: {path}")


def encode_vocab_bin(entries: list[VocabEntry]) -> bytes:
    payload = bytearray()
    payload.extend(_VOCAB_HEADER.pack(b"VOCB", _VERSION, len(entries), _VOCAB_RECORD.size))
    for entry in entries:
        payload.extend(
            _VOCAB_RECORD.pack(
                int(entry.offset),
                int(entry.length),
                int(entry.unigram_count),
                int(entry.doc_freq),
                int(entry.flags),
                0,
            )
        )
    return bytes(payload)


def decode_vocab_bin_bytes(data: bytes, *, error_label: str) -> list[VocabEntry]:
    try:
        magic, version, count, record_size = _VOCAB_HEADER.unpack_from(data, 0)
    except struct.error as exc:
        raise ValueError(error_label) from exc
    if magic != b"VOCB" or version != _VERSION or record_size != _VOCAB_RECORD.size:
        raise ValueError(error_label)
    expected = _VOCAB_HEADER.size + count * _VOCAB_RECORD.size
    if len(data) != expected:
        raise ValueError(error_label)
    entries: list[VocabEntry] = []
    cursor = _VOCAB_HEADER.size
    for _ in range(count):
        offset, length, unigram_count, doc_freq, flags, _reserved = _VOCAB_RECORD.unpack_from(data, cursor)
        entries.append(
            VocabEntry(
                offset=offset,
                length=length,
                unigram_count=unigram_count,
                doc_freq=doc_freq,
                flags=flags,
            )
        )
        cursor += _VOCAB_RECORD.size
    return entries


def write_serve_scores(path: Path, scores: list[int]) -> None:
    path.write_bytes(encode_serve_scores(scores))


def read_serve_scores(path: Path) -> list[int]:
    data = _read_required_bytes(path, "serve scores")
    return decode_serve_scores_bytes(data, error_label=f"Malformed serve scores artifact file: {path}")


def encode_serve_scores(scores: list[int]) -> bytes:
    payload = bytearray()
    payload.extend(_SCORES_HEADER.pack(b"SCRS", _VERSION, len(scores)))
    payload.extend(struct.pack(f"<{len(scores)}h", *scores) if scores else b"")
    return bytes(payload)


def decode_serve_scores_bytes(data: bytes, *, error_label: str) -> list[int]:
    try:
        magic, version, count = _SCORES_HEADER.unpack_from(data, 0)
    except struct.error as exc:
        raise ValueError(error_label) from exc
    if magic != b"SCRS" or version != _VERSION:
        raise ValueError(error_label)
    expected = _SCORES_HEADER.size + count * struct.calcsize("<h")
    if len(data) != expected:
        raise ValueError(error_label)
    if count == 0:
        return []
    return list(struct.unpack_from(f"<{count}h", data, _SCORES_HEADER.size))


def write_token_postings(path: Path, blocks: list[list[TokenPosting]]) -> None:
    path.write_bytes(encode_token_postings(blocks))


def read_token_postings(path: Path) -> list[list[TokenPosting]]:
    data = _read_required_bytes(path, "token postings")
    return decode_token_postings_bytes(data, error_label=f"Malformed token postings artifact file: {path}")


def encode_token_postings(blocks: list[list[TokenPosting]]) -> bytes:
    offsets: list[int] = [0]
    payload = bytearray()
    for block in blocks:
        payload.extend(struct.pack("<H2x", len(block)))
        for item in block:
            payload.extend(_TOKEN_POSTING_RECORD.pack(int(item.token_id), int(item.score_id)))
        offsets.append(len(payload))
    header = _TOKEN_POSTINGS_HEADER.pack(b"TPST", _VERSION, len(blocks), len(offsets))
    offset_blob = struct.pack(f"<{len(offsets)}Q", *offsets)
    return header + offset_blob + bytes(payload)


def decode_token_postings_bytes(data: bytes, *, error_label: str) -> list[list[TokenPosting]]:
    try:
        magic, version, block_count, offset_count = _TOKEN_POSTINGS_HEADER.unpack_from(data, 0)
    except struct.error as exc:
        raise ValueError(error_label) from exc
    if magic != b"TPST" or version != _VERSION or offset_count != block_count + 1:
        raise ValueError(error_label)
    offset_blob_size = offset_count * struct.calcsize("<Q")
    start = _TOKEN_POSTINGS_HEADER.size
    end = start + offset_blob_size
    if len(data) < end:
        raise ValueError(error_label)
    offsets = list(struct.unpack_from(f"<{offset_count}Q", data, start))
    payload = memoryview(data)[end:]
    if offsets[0] != 0 or offsets[-1] != len(payload):
        raise ValueError(error_label)
    blocks: list[list[TokenPosting]] = []
    for left, right in zip(offsets, offsets[1:]):
        segment = payload[left:right]
        if len(segment) < 4:
            raise ValueError(error_label)
        count = struct.unpack_from("<H", segment, 0)[0]
        expected = 4 + count * _TOKEN_POSTING_RECORD.size
        if len(segment) != expected:
            raise ValueError(error_label)
        block: list[TokenPosting] = []
        cursor = 4
        for _ in range(count):
            token_id, score_id = _TOKEN_POSTING_RECORD.unpack_from(segment, cursor)
            block.append(TokenPosting(token_id=token_id, score_id=score_id))
            cursor += _TOKEN_POSTING_RECORD.size
        blocks.append(block)
    return blocks


def write_phrase_lexicon(path: Path, phrases: list[PhraseEntry]) -> None:
    path.write_bytes(encode_phrase_lexicon(phrases))


def read_phrase_lexicon(path: Path) -> list[PhraseEntry]:
    data = _read_required_bytes(path, "phrase lexicon")
    return decode_phrase_lexicon_bytes(data, error_label=f"Malformed phrase lexicon artifact file: {path}")


def encode_phrase_lexicon(phrases: list[PhraseEntry]) -> bytes:
    offsets: list[int] = [0]
    payload = bytearray()
    for phrase in phrases:
        payload.extend(
            _PHRASE_ENTRY_HEADER.pack(
                len(phrase.token_ids),
                int(phrase.display_length_chars),
                int(phrase.flags),
                0,
            )
        )
        payload.extend(struct.pack(f"<{len(phrase.token_ids)}I", *phrase.token_ids))
        offsets.append(len(payload))
    header = _PHRASE_HEADER.pack(b"PHLX", _VERSION, len(phrases), len(offsets))
    offset_blob = struct.pack(f"<{len(offsets)}Q", *offsets)
    return header + offset_blob + bytes(payload)


def decode_phrase_lexicon_bytes(data: bytes, *, error_label: str) -> list[PhraseEntry]:
    try:
        magic, version, phrase_count, offset_count = _PHRASE_HEADER.unpack_from(data, 0)
    except struct.error as exc:
        raise ValueError(error_label) from exc
    if magic != b"PHLX" or version != _VERSION or offset_count != phrase_count + 1:
        raise ValueError(error_label)
    offset_blob_size = offset_count * struct.calcsize("<Q")
    start = _PHRASE_HEADER.size
    end = start + offset_blob_size
    if len(data) < end:
        raise ValueError(error_label)
    offsets = list(struct.unpack_from(f"<{offset_count}Q", data, start))
    payload = memoryview(data)[end:]
    if offsets[0] != 0 or offsets[-1] != len(payload):
        raise ValueError(error_label)
    out: list[PhraseEntry] = []
    for phrase_id, (left, right) in enumerate(zip(offsets, offsets[1:])):
        segment = payload[left:right]
        if len(segment) < _PHRASE_ENTRY_HEADER.size:
            raise ValueError(error_label)
        num_tokens, display_length, flags, _reserved = _PHRASE_ENTRY_HEADER.unpack_from(segment, 0)
        expected = _PHRASE_ENTRY_HEADER.size + num_tokens * struct.calcsize("<I")
        if len(segment) != expected:
            raise ValueError(error_label)
        token_ids = tuple(struct.unpack_from(f"<{num_tokens}I", segment, _PHRASE_ENTRY_HEADER.size)) if num_tokens else ()
        out.append(
            PhraseEntry(
                phrase_id=phrase_id,
                token_ids=token_ids,
                display_length_chars=display_length,
                flags=flags,
            )
        )
    return out


def write_context_graph(path: Path, context_edges: dict[tuple[int, ...], list[ContextEdge]]) -> None:
    path.write_bytes(encode_context_graph(context_edges))


def encode_context_graph(context_edges: dict[tuple[int, ...], list[ContextEdge]]) -> bytes:
    keys = sorted(context_edges)
    payload = bytearray()
    nodes: list[tuple[tuple[int, ...], int]] = []
    for key in keys:
        nodes.append((key, len(payload)))
        edges = context_edges[key]
        payload.extend(_CONTEXT_EDGE_HEADER.pack(len(edges)))
        for edge in edges:
            payload.extend(
                _CONTEXT_EDGE_RECORD.pack(
                    int(edge.target_type),
                    0,
                    0,
                    int(edge.target_id),
                    int(edge.score_id),
                )
            )
    header = _CONTEXT_HEADER.pack(b"CTXT", _CONTEXT_GRAPH_VERSION, len(nodes), len(payload))
    node_blob = bytearray()
    for key, offset in nodes:
        if len(key) > _CONTEXT_MAX_KEY_TOKENS:
            raise ValueError(
                f"Context key of length {len(key)} exceeds the binary format maximum of {_CONTEXT_MAX_KEY_TOKENS}. "
                f"Lower BuildConfig.max_context_tokens to at most {_CONTEXT_MAX_KEY_TOKENS}."
            )
        padded = list(key) + [0] * (_CONTEXT_MAX_KEY_TOKENS - len(key))
        node_blob.extend(_CONTEXT_NODE_RECORD.pack(len(key), *padded, offset))
    return header + bytes(node_blob) + bytes(payload)


def read_context_graph(path: Path) -> dict[tuple[int, ...], list[ContextEdge]]:
    data = _read_required_bytes(path, "context graph")
    return decode_context_graph_bytes(data, error_label=f"Malformed context graph artifact file: {path}")


def decode_context_graph_bytes(data: bytes, *, error_label: str) -> dict[tuple[int, ...], list[ContextEdge]]:
    try:
        magic, version, node_count, payload_size = _CONTEXT_HEADER.unpack_from(data, 0)
    except struct.error as exc:
        raise ValueError(error_label) from exc
    if magic != b"CTXT" or version not in {_VERSION, _CONTEXT_GRAPH_VERSION}:
        raise ValueError(error_label)
    node_record = _CONTEXT_NODE_RECORD_V2 if version == _VERSION else _CONTEXT_NODE_RECORD
    max_key_tokens = 3 if version == _VERSION else _CONTEXT_MAX_KEY_TOKENS
    node_blob_size = node_count * node_record.size
    start = _CONTEXT_HEADER.size
    end = start + node_blob_size
    if len(data) != end + payload_size:
        raise ValueError(error_label)
    payload = memoryview(data)[end:]
    nodes: list[tuple[tuple[int, ...], int]] = []
    cursor = start
    for _ in range(node_count):
        unpacked = node_record.unpack_from(data, cursor)
        key_len = int(unpacked[0])
        token_ids = unpacked[1 : 1 + max_key_tokens]
        offset = int(unpacked[-1])
        if key_len > max_key_tokens:
            raise ValueError(error_label)
        key = tuple(int(item) for item in token_ids[:key_len])
        nodes.append((key, offset))
        cursor += node_record.size
    nodes.append(((), len(payload)))
    out: dict[tuple[int, ...], list[ContextEdge]] = {}
    for index in range(node_count):
        key, left = nodes[index]
        right = nodes[index + 1][1]
        if right < left or right > len(payload):
            raise ValueError(error_label)
        segment = payload[left:right]
        if len(segment) < _CONTEXT_EDGE_HEADER.size:
            raise ValueError(error_label)
        edge_count = _CONTEXT_EDGE_HEADER.unpack_from(segment, 0)[0]
        expected = _CONTEXT_EDGE_HEADER.size + edge_count * _CONTEXT_EDGE_RECORD.size
        if len(segment) != expected:
            raise ValueError(error_label)
        edges: list[ContextEdge] = []
        edge_cursor = _CONTEXT_EDGE_HEADER.size
        for _ in range(edge_count):
            edge_type, _pad1, _pad2, target_id, score_id = _CONTEXT_EDGE_RECORD.unpack_from(segment, edge_cursor)
            edges.append(ContextEdge(target_id=target_id, target_type=edge_type, score_id=score_id))
            edge_cursor += _CONTEXT_EDGE_RECORD.size
        out[key] = edges
    return out


def write_scorer_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(encode_scorer_payload_text(payload), encoding="utf-8")


def read_scorer_payload(path: Path) -> dict[str, Any]:
    try:
        return decode_scorer_payload_text(path.read_text(encoding="utf-8"), error_label=f"Malformed scorer artifact file: {path}")
    except FileNotFoundError:
        return {}


def encode_scorer_payload_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def decode_scorer_payload_text(raw: str, *, error_label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(error_label) from exc
    if not isinstance(payload, dict):
        raise ValueError(error_label)
    return payload


def load_binary_index(base_dir: Path, *, prefix_to_block: dict[str, int]) -> BinaryIndexData:
    manifest = read_manifest(base_dir / MANIFEST_FILENAME)
    files = manifest["files"]
    return BinaryIndexData(
        vocab_strings=read_vocab_strings(base_dir / files["vocab_strings"]),
        vocab_entries=read_vocab_bin(base_dir / files["vocab_bin"]),
        serve_scores=read_serve_scores(base_dir / files["serve_scores"]),
        token_postings=read_token_postings(base_dir / files["token_postings"]),
        phrase_entries=read_phrase_lexicon(base_dir / files["phrase_lexicon"]),
        context_edges=read_context_graph(base_dir / files["context_graph"]),
        prefix_to_block=prefix_to_block,
        scorer_payload=read_scorer_payload(base_dir / files.get("scorer", SCORER_FILENAME)),
        metadata=manifest,
    )


def _read_required_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {label} artifact file: {path}") from exc
