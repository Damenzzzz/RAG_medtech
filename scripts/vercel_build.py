"""Restore large RAG assets that are split for Vercel's source upload limit."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".vercel-build-assets"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            member_path = (destination / member.name).resolve()
            if resolved_destination not in member_path.parents and member_path != resolved_destination:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(destination)


def _restore_split_archive(parts_pattern: str, destination: Path, target: Path) -> None:
    if target.exists():
        return

    parts = sorted(ASSETS.glob(parts_pattern))
    if not parts:
        raise RuntimeError(f"Missing build assets: {parts_pattern}")

    combined = ASSETS / "combined.tar.gz"
    try:
        with combined.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)
        _safe_extract(combined, destination)
    finally:
        combined.unlink(missing_ok=True)


def main() -> None:
    _restore_split_archive(
        "chroma.sqlite3.tar.gz.part-*",
        ROOT / "basic_rag" / "chroma_db",
        ROOT / "basic_rag" / "chroma_db" / "chroma.sqlite3",
    )

    corpus_target = ROOT / "data" / "corpus" / "merged_protocols_structured_clean.jsonl"
    if not corpus_target.exists():
        _safe_extract(ASSETS / "corpus.tar.gz", corpus_target.parent)

    print("Vercel RAG assets restored")


if __name__ == "__main__":
    main()
