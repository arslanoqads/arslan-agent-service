import os
from pathlib import Path

RAW_DIR = Path("data/raw")
DEFAULT_PDFS = ("resume.pdf", "bio.pdf")


def ensure_rag_pdfs(filenames: tuple[str, ...] = DEFAULT_PDFS) -> list[str]:
    """
    Ensure RAG PDFs exist locally.
    - Local/dev: use files already in data/raw/
    - Cloud Run: if RAG_GCS_BUCKET is set, download missing files from that private bucket
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = [RAW_DIR / name for name in filenames]
    missing = [p for p in paths if not p.exists()]

    if not missing:
        return [str(p) for p in paths]

    bucket_name = os.getenv("RAG_GCS_BUCKET")
    if not bucket_name:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing PDF(s): {missing_names}. "
            "Add them under data/raw/ locally, or set RAG_GCS_BUCKET for Cloud Run."
        )

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in missing:
        blob = bucket.blob(path.name)
        if not blob.exists():
            raise FileNotFoundError(
                f"gs://{bucket_name}/{path.name} not found. "
                "Upload the PDF to the private bucket first."
            )
        blob.download_to_filename(str(path))

    return [str(p) for p in paths]
