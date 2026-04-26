from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from importlib.util import find_spec
from unittest.mock import patch

import query_autocomplete
from query_autocomplete.__main__ import main as cli_main
from tests.test_support import fake_marisa_trie_module


class PackagingSmokeTests(unittest.TestCase):
    def test_public_api_is_importable(self) -> None:
        self.assertTrue(hasattr(query_autocomplete, "Autocomplete"))
        self.assertTrue(hasattr(query_autocomplete, "AdaptiveAutocomplete"))
        self.assertTrue(hasattr(query_autocomplete, "AdaptiveStore"))
        self.assertTrue(hasattr(query_autocomplete, "Document"))

    def test_package_metadata_has_base_file_readers_and_chunking_extra(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "python-package" / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn('"pypdf"', text)
        self.assertIn('"python-docx"', text)
        self.assertIn("chunking = [", text)

    def test_python_m_cli_builds_artifact_directory(self) -> None:
        if find_spec("marisa_trie") is None:
            self.skipTest("marisa_trie is not installed in this environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus = Path(tmpdir) / "corpus.txt"
            output = Path(tmpdir) / "artifact"
            corpus.write_text("how to build a deck\nhow to build a desk\n", encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, "-m", "query_autocomplete", "build", "--input", str(corpus), "--output", str(output), "--max-generated-words", "4"],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "token_postings.bin").exists())

    def test_cli_main_builds_artifact_with_fake_marisa(self) -> None:
        with patch.dict(sys.modules, {"marisa_trie": fake_marisa_trie_module()}):
            with tempfile.TemporaryDirectory() as tmpdir:
                corpus = Path(tmpdir) / "corpus.txt"
                output = Path(tmpdir) / "artifact"
                corpus.write_text("how to build a deck\nhow to build a desk\n", encoding="utf-8")
                rc = cli_main(["build", "--input", str(corpus), "--output", str(output), "--max-generated-words", "4"])
                self.assertEqual(rc, 0)
                self.assertTrue((output / "manifest.json").exists())
                self.assertTrue((output / "token_postings.bin").exists())
