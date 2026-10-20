"""Atomic file I/O (plain and gzip), JSON / JSONL helpers, done markers, hashing and logging.

Ported from SSPO v1 (atomic_io.py).

Every write goes to a hidden temporary file in the target directory and is moved into place with ``os.replace``,
so readers never observe a partially written file and a killed writer leaves the previous version intact.
Gzip output uses ``mtime=0`` so identical content always produces identical bytes (stable sha256 manifests).
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import IO, Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

PathLike = str | os.PathLike


# ---------------------------------------------------------------- hashing


def sha1_16(data: str | bytes) -> str:
    """First 16 hex digits of the SHA-1 of ``data`` (str is UTF-8 encoded with replacement)."""
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    return hashlib.sha1(data).hexdigest()[:16]


def sha256_hex(data: str | bytes) -> str:
    """Full SHA-256 hex digest of ``data``."""
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: PathLike, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- paths and logging


def to_rel_path(path: PathLike, root: PathLike | None = None) -> str:
    """POSIX path of ``path`` relative to ``root`` (default: repository root); the base name if outside it."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    base = Path(root or REPO_ROOT).resolve()
    try:
        return p.resolve().relative_to(base).as_posix()
    except ValueError:
        return p.name


def get_logger(name: str = "selectrl") -> logging.Logger:
    """Logger writing ``HH:MM:SS name level message`` lines to stderr (configured once per name)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("SELECTRL_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def log(*parts: Any) -> None:
    """Log the space-joined ``parts`` at INFO level on the ``selectrl`` logger."""
    get_logger().info(" ".join(str(p) for p in parts))


# ---------------------------------------------------------------- atomic writes


def _tmp_path(target: Path) -> Path:
    return target.parent / f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}"


@contextlib.contextmanager
def atomic_open(target: PathLike, gz: bool = False) -> Iterator[IO[bytes]]:
    """Open a binary file handle whose content replaces ``target`` atomically when the block exits cleanly.

    With ``gz`` the handle transparently gzip-compresses (deterministic header, ``mtime=0``).
    On an exception the temporary file is removed and ``target`` is left untouched.
    """
    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    try:
        with open(tmp, "wb") as raw:
            if gz:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zf:
                    yield zf  # type: ignore[misc]
            else:
                yield raw
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def atomic_write_bytes(target: PathLike, data: bytes) -> None:
    """Atomically write ``data`` to ``target``."""
    with atomic_open(target) as f:
        f.write(data)


def atomic_write_text(target: PathLike, text: str) -> None:
    """Atomically write UTF-8 ``text`` to ``target``."""
    atomic_write_bytes(target, text.encode("utf-8"))


def _json_bytes(obj: Any, indent: int | None) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=indent) + "\n").encode("utf-8")


def _write_jsonl(f: IO[bytes], rows: Iterable[Any]) -> int:
    n = 0
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n")
        n += 1
    return n


def atomic_write_json(target: PathLike, obj: Any, indent: int | None = 2) -> None:
    """Atomically write ``obj`` as JSON."""
    atomic_write_bytes(target, _json_bytes(obj, indent))


def atomic_write_json_gz(target: PathLike, obj: Any) -> None:
    """Atomically write ``obj`` as compact gzip-compressed JSON."""
    with atomic_open(target, gz=True) as f:
        f.write(_json_bytes(obj, None))


def atomic_write_jsonl(target: PathLike, rows: Iterable[Any]) -> int:
    """Atomically write ``rows`` (consumed lazily) as JSONL; returns the number of rows."""
    with atomic_open(target) as f:
        return _write_jsonl(f, rows)


def atomic_write_jsonl_gz(target: PathLike, rows: Iterable[Any]) -> int:
    """Atomically write ``rows`` (consumed lazily) as gzip-compressed JSONL; returns the number of rows."""
    with atomic_open(target, gz=True) as f:
        return _write_jsonl(f, rows)


# ---------------------------------------------------------------- reads


def read_json(target: PathLike) -> Any:
    """Load a JSON file."""
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_gz(target: PathLike) -> Any:
    """Load a gzip-compressed JSON file."""
    with gzip.open(target, "rt", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(f: io.TextIOBase) -> Iterator[Any]:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated trailing line of an interrupted non-atomic writer


def iter_jsonl(target: PathLike) -> Iterator[Any]:
    """Iterate the rows of a JSONL file (``.gz`` suffix: gzip), skipping blank or undecodable lines."""
    opener = gzip.open if str(target).endswith(".gz") else open
    with opener(target, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        yield from _iter_jsonl(f)


def read_jsonl(target: PathLike) -> list[Any]:
    """Rows of a plain JSONL file (empty list if the file does not exist)."""
    if not Path(target).exists():
        return []
    with open(target, "r", encoding="utf-8") as f:
        return list(_iter_jsonl(f))


def read_jsonl_gz(target: PathLike) -> list[Any]:
    """Rows of a gzip-compressed JSONL file (empty list if the file does not exist)."""
    if not Path(target).exists():
        return []
    with gzip.open(target, "rt", encoding="utf-8") as f:
        return list(_iter_jsonl(f))


# ---------------------------------------------------------------- done markers


def done_marker_path(stage: PathLike) -> Path:
    """Marker file of a stage: ``<dir>/done``, or ``stage`` itself if it is already named ``done`` / ``*.done``."""
    p = Path(stage)
    if p.name == "done" or p.suffix == ".done":
        return p
    return p / "done"


def has_done_marker(stage: PathLike) -> bool:
    """True if the stage's done marker exists."""
    return done_marker_path(stage).exists()


def write_done_marker(stage: PathLike, info: dict[str, Any] | None = None) -> Path:
    """Atomically write the stage's done marker (UTC completion time plus ``info``)."""
    marker = done_marker_path(stage)
    atomic_write_json(
        marker, {"status": "done", "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(info or {})}
    )
    return marker
