"""Remote dataset acquisition.

The dataset for this project is pulled from a public URL at runtime rather than
being read straight off disk. Three things make that safe enough to deploy:

  1. **Integrity checking.** The upstream file's SHA-256 is recorded here. A
     download whose digest does not match is rejected rather than silently
     trained on — remote files are edited and replaced without notice, and a
     changed dataset that still parses is the kind of failure that produces
     wrong numbers instead of an error.

  2. **Local caching.** A verified download is written to `data/.cache/` and
     reused on subsequent runs, so the network is touched once rather than on
     every app restart.

  3. **A committed snapshot as fallback.** `data/soybean.csv` is the same data,
     checked into the repository. If the network is unavailable, the host is rate
     limiting, or the digest fails, the loader falls back to the snapshot and
     reports which source it used.

That last point is the important one. Fetching at runtime is, on its own, a
reliability *regression* for a deployed app: a hosting platform that cannot reach
the upstream host gets a dead app instead of a working one. During development of
this project the upstream host returned HTTP 429 for an extended period, which is
exactly the scenario the fallback exists for. The remote path is the convenience;
the committed snapshot is what makes the deployment dependable.

Only the standard library is used for the transfer — `urllib.request` — so this
adds no dependency to requirements.txt.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
CACHE_DIR = os.path.join(DATA_DIR, ".cache")

DOWNLOAD_TIMEOUT = 20  # seconds
USER_AGENT = "ml-assignment-2/1.0"


@dataclass(frozen=True)
class RemoteSource:
    """A dataset available over HTTP, with a local snapshot to fall back to."""

    name: str
    url: str
    sha256: str
    target_col: str
    snapshot: str
    description: str
    rename: dict | None = None      # upstream column name -> project column name

    @property
    def cache_path(self) -> str:
        return os.path.join(CACHE_DIR, f"{self.name}.csv")

    @property
    def snapshot_path(self) -> str:
        return os.path.join(DATA_DIR, self.snapshot)


SOYBEAN = RemoteSource(
    name="soybean",
    url="https://raw.githubusercontent.com/selva86/datasets/master/Soybean.csv",
    sha256="96eac8047b2034523c57c5d6ae32ecc64ba2da139b1d930a838a35de3cda5ce6",
    target_col="disease",
    snapshot="soybean.csv",
    description=(
        "UCI Soybean (Large): 683 field observations of diseased soybean plants, "
        "35 categorical observations each, classified into 19 diseases."
    ),
    # The upstream file uses dots in column names and calls the target "Class".
    # Renaming happens at load time rather than by editing the file, so the
    # committed snapshot stays byte-identical to upstream and the same SHA-256
    # verifies both.
    rename={"Class": "disease"},
)

DEFAULT_SOURCE = SOYBEAN


class IntegrityError(RuntimeError):
    """Raised when a downloaded file's digest does not match the expected value."""


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(url: str, timeout: int = DOWNLOAD_TIMEOUT) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout,
                                context=context) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(url, response.status, "unexpected status",
                                         response.headers, None)
        return response.read()


def acquire(source: RemoteSource = DEFAULT_SOURCE,
            allow_network: bool = True,
            force_refresh: bool = False) -> tuple[str, str]:
    """Make the dataset available locally.

    Returns (path, origin) where origin is one of "cache", "download" or
    "snapshot", so the caller can tell the user where the data actually came
    from rather than leaving it ambiguous.

    Never raises on a network problem — it degrades to the committed snapshot.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(source.cache_path) and not force_refresh:
        with open(source.cache_path, "rb") as handle:
            if digest(handle.read()) == source.sha256:
                return source.cache_path, "cache"
        os.remove(source.cache_path)          # corrupt or stale; re-fetch

    if allow_network:
        try:
            payload = _download(source.url)
            found = digest(payload)
            if found != source.sha256:
                raise IntegrityError(
                    f"Digest mismatch for {source.url}\n"
                    f"  expected {source.sha256}\n"
                    f"  received {found}\n"
                    "The upstream file has changed. Falling back to the "
                    "committed snapshot; update the recorded digest only after "
                    "reviewing what changed."
                )
            with open(source.cache_path, "wb") as handle:
                handle.write(payload)
            return source.cache_path, "download"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                IntegrityError):
            pass                              # fall through to the snapshot

    if os.path.exists(source.snapshot_path):
        return source.snapshot_path, "snapshot"

    raise FileNotFoundError(
        f"Could not reach {source.url} and no snapshot exists at "
        f"{source.snapshot_path}."
    )


def verify_snapshot(source: RemoteSource = DEFAULT_SOURCE) -> bool:
    """True if the committed snapshot matches the recorded upstream digest."""
    if not os.path.exists(source.snapshot_path):
        return False
    with open(source.snapshot_path, "rb") as handle:
        return digest(handle.read()) == source.sha256


def describe_origin(origin: str) -> str:
    return {
        "download": "downloaded from the upstream URL and digest-verified",
        "cache": "read from the local verified cache",
        "snapshot": "read from the committed snapshot (network unavailable "
                    "or digest mismatch)",
    }.get(origin, origin)
