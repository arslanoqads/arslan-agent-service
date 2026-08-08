import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

class ResumeRAGEngine:
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Cannot find resume at path: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        self.vector_store = InMemoryVectorStore.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings()
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def query_resume(self, query: str) -> str:
        results = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])
