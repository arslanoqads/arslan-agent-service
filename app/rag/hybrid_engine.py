import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.context.budget import sandwich
from app.rag.corpus import current_bio, discover_documents, latest_resume, wants_older_resumes

INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", "data/index/manifest.json"))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class HybridRAGEngine:
    def __init__(self):
        self.documents = discover_documents()
        if not self.documents:
            raise FileNotFoundError("No resume or biography PDFs found in data/raw/.")
        self.chunks = self._load_or_embed()
        self._bm25 = BM25Retriever.from_texts(
            [chunk["text"] for chunk in self.chunks],
            metadatas=[chunk["metadata"] for chunk in self.chunks],
        )
        self._bm25.k = 8

    def _load_or_embed(self) -> list[dict]:
        stored = self._read_index()
        stored_hashes = {item["path"]: item["sha256"] for item in stored.get("files", [])}
        current_hashes = {item["path"]: item["sha256"] for item in self.documents}
        if stored.get("chunks") and stored_hashes == current_hashes:
            return stored["chunks"]

        reusable = []
        if stored.get("chunks"):
            reusable = [
                chunk
                for chunk in stored["chunks"]
                if stored_hashes.get(chunk["metadata"]["path"]) == current_hashes.get(chunk["metadata"]["path"])
            ]
        reusable_paths = {chunk["metadata"]["path"] for chunk in reusable}
        fresh_docs = [item for item in self.documents if item["path"] not in reusable_paths]
        embedded = self._embed_documents(fresh_docs) if fresh_docs else []
        chunks = reusable + embedded
        self._write_index(chunks)
        return chunks

    def _embed_documents(self, documents: list[dict]) -> list[dict]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        pieces = []
        for document in documents:
            for page_index, page in enumerate(PyPDFLoader(document["path"]).load(), start=1):
                for split in splitter.split_documents([page]):
                    page_number = split.metadata.get("page", page_index - 1) + 1
                    pieces.append(
                        {
                            "text": split.page_content,
                            "metadata": {
                                "doc_type": document["doc_type"],
                                "version": document["version"],
                                "filename": document["filename"],
                                "path": document["path"],
                                "page": page_number,
                            },
                        }
                    )
        if not pieces:
            return []
        from langchain_openai import OpenAIEmbeddings

        vectors = OpenAIEmbeddings().embed_documents([piece["text"] for piece in pieces])
        for piece, vector in zip(pieces, vectors):
            piece["embedding"] = vector
        return pieces

    def _read_index(self) -> dict:
        if not INDEX_PATH.exists():
            return {}
        return json.loads(INDEX_PATH.read_text())

    def _write_index(self, chunks: list[dict]) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "files": [
                {"path": item["path"], "sha256": item["sha256"], "doc_type": item["doc_type"], "version": item["version"]}
                for item in self.documents
            ],
            "chunks": chunks,
        }
        INDEX_PATH.write_text(json.dumps(payload))

    def _allowed(self, query_text: str) -> set[tuple[str, int]]:
        allowed = set()
        if wants_older_resumes(query_text):
            allowed.update((item["doc_type"], item["version"]) for item in self.documents if item["doc_type"] == "resume")
        else:
            latest = latest_resume()
            if latest:
                allowed.add(("resume", latest["version"]))
        bio = current_bio()
        if bio:
            allowed.add(("bio", bio["version"]))
        return allowed

    def retrieve_chunks(self, query_text: str, *, job_match: bool = False) -> list[dict]:
        """Two-step retrieval: BM25 narrows candidates, dense scoring picks the top ones."""
        allowed = self._allowed(query_text)
        if job_match:
            latest = latest_resume()
            allowed = {("resume", latest["version"])} if latest else set()

        # Step 1 — keyword / sparse retrieval to narrow the list (IDs, exact phrases).
        candidates: list[dict] = []
        seen: set[str] = set()
        for index, doc in enumerate(self._bm25.invoke(query_text)):
            meta = doc.metadata or {}
            key = (meta.get("doc_type"), meta.get("version"))
            if key not in allowed:
                continue
            text = doc.page_content or ""
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(
                {
                    **meta,
                    "text": text,
                    "bm25_rank": index,
                    "score": 1 - (index * 0.05),
                }
            )
            if len(candidates) >= 8:
                break

        # Fall back to all allowed chunks when BM25 finds nothing useful.
        if not candidates:
            for chunk in self.chunks:
                key = (chunk["metadata"]["doc_type"], chunk["metadata"]["version"])
                if key not in allowed:
                    continue
                candidates.append({**chunk["metadata"], "text": chunk["text"], "embedding": chunk.get("embedding")})

        # Step 2 — dense re-rank of the narrowed list.
        from langchain_openai import OpenAIEmbeddings

        query_vector = OpenAIEmbeddings().embed_query(query_text)
        ranked = []
        for item in candidates:
            embedding = item.get("embedding")
            if embedding is None:
                # Match BM25 hits back to indexed embeddings when present.
                for chunk in self.chunks:
                    if chunk["text"] == item.get("text"):
                        embedding = chunk.get("embedding")
                        break
            dense = _cosine(query_vector, embedding or [])
            bm25_boost = float(item.get("score") or 0) * 0.15
            ranked.append({**{k: v for k, v in item.items() if k != "embedding"}, "score": dense + bm25_boost})
        ranked.sort(key=lambda item: item.get("score", 0), reverse=True)
        return sandwich(ranked)[:3]

    def retrieve(self, query_text: str):
        chunks = self.retrieve_chunks(query_text)
        return [_Doc(chunk["text"], chunk) for chunk in chunks]

    def query(self, query_text: str) -> str:
        chunks = self.retrieve_chunks(query_text)
        if not chunks:
            return "No resume excerpts were retrieved."
        return "\n\n".join(
            f"{chunk['doc_type']} v{chunk['version']}, page {chunk['page']}: {chunk['text']}" for chunk in chunks
        )


class _Doc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata
