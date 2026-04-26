from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import TypeGuard

from query_autocomplete.models import Document

ARTIFACT_VERSION = 1
DEFAULT_ARTIFACT_ROOT = ".query_autocomplete_artifacts"
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def default_artifact_root(base_dir: str | Path | None = None) -> Path:
    anchor = Path.cwd() if base_dir is None else Path(base_dir).expanduser()
    return anchor.resolve() / DEFAULT_ARTIFACT_ROOT


def _is_windows_absolute_path(raw: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(raw))


def is_managed_artifact_name(path: str | Path) -> TypeGuard[str]:
    if not isinstance(path, str):
        return False
    raw = path.strip()
    if not raw:
        return False
    if raw.startswith("."):
        return False
    if "/" in raw or "\\" in raw:
        return False
    if Path(raw).is_absolute() or _is_windows_absolute_path(raw):
        return False
    return True


def resolve_managed_artifact_directory(name: str, *, base_dir: str | Path | None = None) -> Path:
    raw = name.strip()
    if not raw:
        raise ValueError("Artifact name must be non-empty.")
    return default_artifact_root(base_dir) / raw


def make_default_artifact_name(documents: list[Document]) -> str:
    return f"artifact-{secrets.token_urlsafe(6).lower()}"


def reserve_default_artifact_directory(
    documents: list[Document],
    *,
    base_dir: str | Path | None = None,
) -> Path:
    root = default_artifact_root(base_dir)
    stem = make_default_artifact_name(documents)
    candidate = root / stem
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stem}-{suffix}"
        suffix += 1
    return candidate


def resolve_artifact_directory(path: str | Path) -> Path:
    """Directory that holds ``manifest.json`` and sibling binary files.

    * **Absolute path** (including drive letters on Windows): used as-is after
      :meth:`pathlib.Path.resolve`.
    * **Anything else** (a bare name like ``my_index``, ``./out``, or
      ``subdir/index``): resolved under :func:`pathlib.Path.cwd` so the process
      working directory is the anchor for non-absolute paths.

    ``~`` in the path is expanded before resolving.
    """
    p = Path(path).expanduser()
    raw = str(path).strip()
    if not raw:
        raise ValueError("Artifact path must be non-empty.")
    if p.is_absolute() or _is_windows_absolute_path(raw):
        return p.resolve()
    return (Path.cwd() / p).resolve()


def resolve_storage_directory(path: str | Path) -> Path:
    if is_managed_artifact_name(path):
        return resolve_managed_artifact_directory(path)
    return resolve_artifact_directory(path)
