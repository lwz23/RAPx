#!/usr/bin/env python3
"""Compute the deterministic RAP source-tree SHA-256 used by the Linux baseline."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


EXCLUDED_DIRECTORY_NAMES = {".git", "target", "__pycache__"}
CHUNK_SIZE = 1024 * 1024


def source_tree_sha256(root: Path) -> str:
    """Hash a source tree using the frozen RAP provenance algorithm."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    digest = hashlib.sha256()
    paths: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=raise_walk_error
    ):
        base = Path(directory)
        traversable_directories: list[str] = []
        for name in sorted(dirnames):
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            candidate = base / name
            if candidate.is_symlink():
                paths.append(candidate)
            else:
                traversable_directories.append(name)
        dirnames[:] = traversable_directories
        paths.extend(base / name for name in sorted(filenames))

    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            content = os.readlink(path).encode("utf-8")
            digest.update(b"L")
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the frozen RAP source-tree SHA-256."
    )
    parser.add_argument("path", type=Path, help="source-tree directory to hash")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(source_tree_sha256(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
