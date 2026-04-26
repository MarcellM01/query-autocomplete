from __future__ import annotations

from pathlib import Path


def read_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8-sig", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF input requires the 'pypdf' dependency, which is included in the base package."
            ) from exc
        reader = PdfReader(str(path))
        pages = [text for page in reader.pages if (text := page.extract_text())]
        if not pages:
            raise ValueError(f"No extractable text found in PDF file: {path}")
        return "\n".join(pages)
    if suffix == ".docx":
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise ImportError(
                "DOCX input requires the 'python-docx' dependency, which is included in the base package."
            ) from exc
        doc = DocxDocument(str(path))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
        if not text.strip():
            raise ValueError(f"No extractable text found in DOCX file: {path}")
        return text
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: .txt, .pdf, .docx")
