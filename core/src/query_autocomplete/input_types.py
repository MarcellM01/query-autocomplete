from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeAlias

from query_autocomplete.models import Document
from query_autocomplete.preprocessing.file_reader import read_text_from_file


DocumentLike: TypeAlias = Document | str | Path


def coerce_documents(documents: Iterable[DocumentLike]) -> list[Document]:
    coerced: list[Document] = []
    for item in documents:
        if isinstance(item, Document):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append(Document(text=item))
        elif isinstance(item, Path):
            coerced.append(Document(text=read_text_from_file(item), doc_id=item.name))
        else:
            raise TypeError(f"Expected Document, str, or Path, got {type(item).__name__}.")
    return coerced
