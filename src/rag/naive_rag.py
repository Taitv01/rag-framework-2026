"""
Naive RAG
=========

Basic RAG implementation: Vector Search + LLM Generation.
Supports Smart Knowledge Library Ingestion and Automatic Document Classification.

Usage:
    rag = NaiveRAG()
    rag.ingest_to_library("path/to/docs/")
    answer = rag.query("What is Python?")
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager
from src.core.markdown_index import MarkdownFolderIndexer
from src.core.library_manager import LibraryManager


class NaiveRAG:
    """
    Basic RAG implementation with Smart Knowledge Library support.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "huggingface",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_provider: str = "faiss",
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        vector_store_url: Optional[str] = None,
        vector_store_api_key: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        system_prompt: Optional[str] = None,
    ):
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = EmbeddingsManager(
            provider=embedding_provider,
            model_name=embedding_model,
        )
        self.vector_store = VectorStoreManager(
            provider=vector_store_provider,
            embeddings=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
            url=vector_store_url,
            api_key=vector_store_api_key,
        )
        self.llm = LLMManager(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        self.retrieval_k = retrieval_k
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        self._documents = []
        self._chunks = []
        self.library_manager = LibraryManager(persist_directory or "library")

    def _get_default_system_prompt(self) -> str:
        return """You are a helpful AI assistant. Use the provided context to answer the user's question.

Rules:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Be concise and accurate
4. Cite relevant parts of the context when possible

Context:
{context}

Question: {question}"""

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        if isinstance(sources, (str, Path)):
            sources = [sources]

        all_docs = []
        for source in sources:
            source = Path(source)
            if source.is_dir():
                docs = self.document_loader.load_directory(source, metadata=metadata)
            else:
                docs = self.document_loader.load(source, metadata=metadata)
            all_docs.extend(docs)

        self._documents.extend(all_docs)
        chunks = self.text_splitter.split_documents(all_docs)
        self._chunks.extend(chunks)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def ingest_to_library(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        override_category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest, perform OCR if needed, auto-classify into library categories, and index into vector store.

        Args:
            sources: Single path or list of file/directory paths
            override_category: Optional explicit category
            extra_metadata: Extra metadata dictionary

        Returns:
            Dict containing ingestion summary and list of organized records
        """
        if isinstance(sources, (str, Path)):
            sources = [sources]

        files_to_process = []
        for src in sources:
            p = Path(src)
            if p.is_dir():
                files_to_process.extend([f for f in p.rglob("*") if f.is_file() and not f.name.startswith(".")])
            elif p.is_file():
                files_to_process.append(p)

        all_docs = []
        records = []

        for fpath in files_to_process:
            docs, rec = self.library_manager.ingest_and_organize(
                fpath,
                override_category=override_category,
                extra_metadata=extra_metadata,
            )
            all_docs.extend(docs)
            records.append(rec)

        if all_docs:
            self._documents.extend(all_docs)
            chunks = self.text_splitter.split_documents(all_docs)
            self._chunks.extend(chunks)
            self.vector_store.add_documents(chunks)
            num_chunks = len(chunks)
        else:
            num_chunks = 0

        return {
            "status": "success",
            "files_processed": len(records),
            "chunks_indexed": num_chunks,
            "records": records,
        }

    def refresh_markdown_directory(
        self,
        directory: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        strict: bool = False,
    ) -> Dict[str, Any]:
        directory = Path(directory).resolve()
        indexer = MarkdownFolderIndexer()
        result, current_manifest = indexer.compare(
            directory,
            Path(manifest_path) if manifest_path else None,
        )

        if not force and not result.changed:
            return result.to_dict()

        documents = self.document_loader.load_markdown_directory(
            directory,
            metadata=metadata,
            strict=strict,
        )
        chunks = self.text_splitter.split_documents(documents)
        indexer.assign_stable_chunk_ids(chunks)

        self._replace_source_root(str(directory), documents, chunks)
        indexer.save_manifest(Path(result.manifest_path), current_manifest)

        result.documents_loaded = len(documents)
        result.chunks_indexed = len(chunks)
        result.rebuilt = True

        return result.to_dict()

    def _replace_source_root(
        self,
        source_root: str,
        documents: List[Document],
        chunks: List[Document],
    ) -> None:
        remaining_documents = [
            doc
            for doc in self._documents
            if doc.metadata.get("source_root") != source_root
        ]
        remaining_chunks = [
            chunk
            for chunk in self._chunks
            if chunk.metadata.get("source_root") != source_root
        ]

        provider = getattr(self.vector_store.config, "provider", "faiss")
        self._documents = remaining_documents + documents
        self._chunks = remaining_chunks + chunks

        if provider == "faiss":
            self._rebuild_vector_store_from_chunks()
            return

        self.vector_store.delete(filter={"source_root": source_root})
        self._add_chunks_to_vector_store(chunks)

    def _rebuild_vector_store_from_chunks(self) -> None:
        config = self.vector_store.config
        self.vector_store = VectorStoreManager(
            provider=config.provider,
            embeddings=self.embeddings,
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            url=config.url,
            api_key=config.api_key,
        )
        self._add_chunks_to_vector_store(self._chunks)

    def _add_chunks_to_vector_store(self, chunks: List[Document]) -> None:
        if not chunks:
            return

        chunk_ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
        if all(chunk_ids):
            self.vector_store.add_documents(chunks, ids=chunk_ids)
        else:
            self.vector_store.add_documents(chunks)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        docs = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas or [{}] * len(texts))
        ]
        self._documents.extend(docs)
        chunks = self.text_splitter.split_documents(docs)
        self._chunks.extend(chunks)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        k = k or self.retrieval_k
        return self.vector_store.similarity_search(query, k=k)

    def query(
        self,
        question: str,
        k: Optional[int] = None
    ) -> str:
        docs = self.retrieve(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.system_prompt.format(context=context, question=question)
        response = self.llm.generate(prompt)
        return response

    def query_with_sources(
        self,
        question: str,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        docs = self.retrieve(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.system_prompt.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}

    def clear(self) -> None:
        self.vector_store.clear()
        self._documents = []
        self._chunks = []

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
