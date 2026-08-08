import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings


class HybridRAGEngine:
    """
    Hybrid RAG Engine combining Dense Vector Search (OpenAI Embeddings) 
    and Sparse Keyword Search (BM25) over multiple PDF files.
    """
    def __init__(self, pdf_paths: list[str]):
        all_docs = []

        # 1. Load all provided PDF documents
        for path in pdf_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Cannot find document at path: {path}")
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())

        # 2. Chunk documents into smaller text snippets
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(all_docs)

        # 3. Dense Retriever (Cosine Similarity via OpenAI Embeddings)
        vector_store = InMemoryVectorStore.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 4. Sparse Retriever (BM25 Keyword Matcher)
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3

        # 5. Hybrid Ensemble Retriever (50% BM25 + 50% Vector Embeddings)
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )

    def query(self, query_text: str) -> str:
        """Executes a hybrid search and returns combined matching context."""
        results = self.retriever.invoke(query_text)
        return "\n\n".join([doc.page_content for doc in results])
