"""Wikitext for benchmarks: a single ``all.txt`` under ``benchmarks/assets/`` (or a custom dir)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
ALL_TXT = "all.txt"
DEFAULT_WIKITEXT_DATA_DIR = Path(__file__).resolve().parent / "assets" / "wikitext-2-raw-v1"


def _all_txt_path(out_dir: Path) -> Path:
    return Path(out_dir) / ALL_TXT


def _write_all_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.rstrip("\n"))
            f.write("\n")


def _collect_split_lines(ds_split) -> list[str]:
    return [str(x).rstrip("\n") for x in ds_split["text"]]


def wikitext_all_txt_exists(out_dir: Path) -> bool:
    return _all_txt_path(out_dir).is_file()


def download_wikitext_all_txt(
    out_dir: Path,
    *,
    dataset: str = DATASET,
    config: str = CONFIG,
) -> None:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Install the `datasets` package:  python -m pip install datasets") from e
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset, config)
    merged: list[str] = []
    for hf_name in ("train", "validation", "test"):
        if hf_name in ds:
            merged.extend(_collect_split_lines(ds[hf_name]))
    if not merged:
        raise ValueError("Empty dataset from Hugging Face (unexpected for wikitext).")
    _write_all_txt(_all_txt_path(out_dir), merged)


def get_wikitext_data(
    out_dir: Path,
    *,
    source: Literal["auto", "export", "hf"] = "auto",
    dataset: str = DATASET,
    config: str = CONFIG,
) -> list[str]:
    out_dir = Path(out_dir)

    def from_all_txt() -> list[str] | None:
        path = _all_txt_path(out_dir)
        if path.is_file():
            return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
        return None

    def from_hf() -> list[str]:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Install the `datasets` package to load wikitext from Hugging Face.") from e
        ds = load_dataset(dataset, config)
        merged: list[str] = []
        for hf_name in ("train", "validation", "test"):
            if hf_name in ds:
                merged.extend(_collect_split_lines(ds[hf_name]))
        if not merged:
            raise ValueError("Empty dataset from Hugging Face (unexpected for wikitext).")
        return merged

    if source == "export":
        out = from_all_txt()
        if out is None:
            raise FileNotFoundError(f"Missing {ALL_TXT!r} under {out_dir}.")
        return out
    if source == "hf":
        return from_hf()
    if source == "auto":
        out = from_all_txt()
        if out is not None:
            return out
        return from_hf()
    msg = f"unknown source: {source!r}"
    raise ValueError(msg)
