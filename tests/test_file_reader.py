from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from query_autocomplete.preprocessing.file_reader import read_text_from_file


class FileReaderTests(unittest.TestCase):
    def test_pdf_without_extractable_text_raises_value_error(self) -> None:
        fake_reader = Mock()
        fake_reader.pages = [Mock(extract_text=Mock(return_value=None)), Mock(extract_text=Mock(return_value=""))]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.pdf"
            path.write_bytes(b"%PDF")
            fake_pypdf = types.SimpleNamespace(PdfReader=Mock(return_value=fake_reader))
            with patch.dict("sys.modules", {"pypdf": fake_pypdf}):
                with self.assertRaisesRegex(ValueError, "No extractable text"):
                    read_text_from_file(path)

    def test_docx_without_extractable_text_raises_value_error(self) -> None:
        fake_doc = Mock()
        fake_doc.paragraphs = [Mock(text=""), Mock(text="   ")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.docx"
            path.write_bytes(b"docx")
            fake_docx = types.SimpleNamespace(Document=Mock(return_value=fake_doc))
            with patch.dict("sys.modules", {"docx": fake_docx}):
                with self.assertRaisesRegex(ValueError, "No extractable text"):
                    read_text_from_file(path)


if __name__ == "__main__":
    unittest.main()
