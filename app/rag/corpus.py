import hashlib
import os
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
VERSION_RE = re.compile(r"v(\d+)", re.IGNORECASE)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_from_name(path: Path, default: int) -> int:
    match = VERSION_RE.search(path.stem)
    return int(match.group(1)) if match else default


def _sync_gcs() -> None:
    bucket_name = os.getenv("RAG_GCS_BUCKET")
    if not bucket_name:
        return
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in bucket.list_blobs():
        name = blob.name
        if not name.lower().endswith(".pdf"):
            continue
        if name.startswith("resumes/") or name.startswith("bio/") or name in {"resume.pdf", "bio.pdf"}:
            dest = RAW_DIR / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                blob.download_to_filename(str(dest))


def discover_documents(root: Path | None = None) -> list[dict]:
    raw = root or RAW_DIR
    if root is None:
        _sync_gcs()
    raw.mkdir(parents=True, exist_ok=True)
    found: list[dict] = []

    legacy_resume = raw / "resume.pdf"
    if legacy_resume.exists():
        found.append(_record("resume", 1, legacy_resume))
    for path in sorted((raw / "resumes").glob("*.pdf")) if (raw / "resumes").exists() else []:
        version = _version_from_name(path, len([item for item in found if item["doc_type"] == "resume"]) + 1)
        if any(item["doc_type"] == "resume" and item["version"] == version for item in found):
            continue
        found.append(_record("resume", version, path))

    legacy_bio = raw / "bio.pdf"
    if legacy_bio.exists():
        found.append(_record("bio", 1, legacy_bio))
    for path in sorted((raw / "bio").glob("*.pdf")) if (raw / "bio").exists() else []:
        version = _version_from_name(path, len([item for item in found if item["doc_type"] == "bio"]) + 1)
        if any(item["doc_type"] == "bio" and item["version"] == version for item in found):
            continue
        found.append(_record("bio", version, path))

    return sorted(found, key=lambda item: (item["doc_type"], item["version"]))


def _record(doc_type: str, version: int, path: Path) -> dict:
    return {
        "doc_type": doc_type,
        "version": version,
        "path": str(path),
        "filename": path.name,
        "sha256": file_hash(path),
    }


def latest_resume() -> dict | None:
    resumes = [item for item in discover_documents() if item["doc_type"] == "resume"]
    return resumes[-1] if resumes else None


def current_bio() -> dict | None:
    bios = [item for item in discover_documents() if item["doc_type"] == "bio"]
    return bios[-1] if bios else None


def corpus_fingerprint() -> str:
    parts = [f"{item['doc_type']}:{item['version']}:{item['sha256']}" for item in discover_documents()]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def wants_older_resumes(query: str) -> bool:
    text = (query or "").lower()
    return any(word in text for word in ("earlier", "previous", "older", "old resume", "version 1", "v1"))
