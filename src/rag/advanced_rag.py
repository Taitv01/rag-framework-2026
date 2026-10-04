"""
Advanced RAG
============

Enhanced RAG with hybrid search, re-ranking, and query transformation.

Features:
- Hybrid search (vector + BM25) with Vietnamese tokenization
- Cross-encoder re-ranking (Vietnamese-aware)
- Query transformation (bilingual)
- Document grading
- Semantic caching (embedding-based similarity)
- Contextual retrieval (Anthropic pattern)
- HyDE (Hypothetical Document Embeddings)
- Multi-query with RRF (Reciprocal Rank Fusion)
- Streaming responses

Pipeline (one path for query, query_detailed and stream):
1. Load and chunk documents (optionally with contextual headers)
2. Create vector + BM25 indices
3. Check semantic cache
4. Rewrite the query (by default only when typed without diacritics)
5. Hybrid retrieval of child chunks, reranked as parent passages
6. Optional relevance grading (one batched LLM call, or a reranker cut-off)
7. No relevant context: answer "not enough information" (or web fallback)
8. Generate the answer with the LLM (usually the only LLM call)
9. Keep only the sources the answer cites; cache the result

Usage:
    rag = AdvancedRAG(use_cache=True, use_contextual_chunking=True)
    rag.add_documents(["docs/"])
    answer = rag.query("Thạch Sanh là ai?")
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Generator
from pathlib import Path

from langchain_core.documents import Document

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.retriever import RetrieverManager, has_diacritics
from src.core.llm import LLMManager
from src.core.markdown_index import MarkdownFolderIndexer
from src.monitoring.tracing import record_steps, step
from src.utils.rwlock import ReadWriteLock
from src.rag.multimodal import build_multimodal_prompt

logger = logging.getLogger(__name__)

# Answer given without calling the LLM when no relevant context was found.
NO_CONTEXT_ANSWER = (
    "Tôi không có đủ thông tin trong tài liệu để trả lời câu hỏi này. / "
    "I don't have enough information in the documents to answer this question."
)

QUERY_REWRITE_MODES = ("auto", "always", "never")
GRADING_MODES = ("none", "llm", "reranker")


def _trace_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """What a trace shows of retrieved passages: source and score, not the text."""
    return [
        {
            "source": (doc.metadata or {}).get("source"),
            "score": (doc.metadata or {}).get("relevance_score"),
        }
        for doc in docs
    ]


@dataclass
class PreparedQuery:
    """Everything decided before generation; shared by query, query_detailed and stream."""
    question: str
    search_query: str
    retrieved: List[Document] = field(default_factory=list)
    docs: List[Document] = field(default_factory=list)
    web_docs: List[Document] = field(default_factory=list)
    prompt: Optional[str] = None
    answer: Optional[str] = None  # set when no LLM call is needed (cache hit, no context)
    cache_hit: bool = False
    abstained: bool = False
    cache_scope: Optional[str] = None  # cached answers are only reused in the same scope


class AdvancedRAG:
    """
    Advanced RAG with hybrid search and re-ranking.

    Implements a sophisticated retrieval pipeline with:
    - Query transformation for better recall
    - Hybrid search (vector + BM25)
    - Cross-encoder re-ranking for precision
    - Document relevance grading

    Example:
        rag = AdvancedRAG(
            llm_provider="openai",
            use_hybrid=True,
            use_reranking=True
        )

        rag.add_documents(["documents/"])

        # Standard query
        answer = rag.query("Thạch Sanh là ai?")

        # Query with detailed results
        result = rag.query_detailed("Thạch Sanh đánh đại bàng như thế nào?")
        print(result["answer"])
        print(result["transformed_query"])
        print(result["relevant_docs"])
    """

    # Defaults for instances built without __init__ (tests, subclasses).
    use_parent_context = True
    parent_fanout = 3
    query_rewrite = "auto"
    grading = "none"
    min_relevance_score = None
    _cache = None
    _web_searcher = None
    _hallucination_grader = None
    use_hyde = False
    use_multi_query_rrf = False

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "huggingface",
        embedding_model: Optional[str] = None,
        embedding_device: Optional[str] = None,
        vector_store_provider: str = "faiss",
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        vector_store_url: Optional[str] = None,
        vector_store_api_key: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_parent_context: bool = True,
        parent_fanout: int = 3,
        reranker=None,
        reranker_model: Optional[str] = None,
        query_rewrite: str = "auto",
        grading: str = "none",
        min_relevance_score: Optional[float] = None,
        system_prompt: Optional[str] = None,
        # Phase 2 options
        use_cache: bool = False,
        cache_ttl: int = 3600,
        cache_threshold: float = 0.95,
        use_contextual_chunking: bool = False,
        use_hyde: bool = False,
        use_multi_query_rrf: bool = False,
        num_query_variations: int = 3,
        # Phase 3 options
        use_web_search: bool = False,
        web_search_provider: str = "duckduckgo",
        web_search_api_key: Optional[str] = None,
        use_hallucination_check: bool = False,
        hallucination_threshold: float = 0.8,
        use_metadata_enhancement: bool = False,
    ):
        """
        Initialize Advanced RAG.

        Args:
            llm_provider: LLM provider
            llm_model: LLM model name (default: the provider's default model)
            llm_api_key: LLM API key
            embedding_provider: Embedding provider
            embedding_model: Embedding model name (default: the provider's default,
                BAAI/bge-m3 for huggingface)
            embedding_device: Device for local embedding models ('cpu', 'cuda')
            vector_store_provider: Vector store provider
            collection_name: Vector store collection/index name
            persist_directory: Directory for persistent local vector stores
            vector_store_url: URL for remote vector stores
            vector_store_api_key: API key for remote vector stores
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            retrieval_k: Number of documents to retrieve
            use_hybrid: Enable hybrid search
            use_reranking: Enable re-ranking
            use_parent_context: Search small child chunks but return their
                parent chunks, so the LLM sees the surrounding passage
            parent_fanout: Child chunks searched per requested parent
            reranker: An already loaded cross-encoder to use (e.g. shared
                between pipelines) instead of loading one
            reranker_model: Name of that cross-encoder
            query_rewrite: LLM query rewrite before retrieval: "auto" (only for
                queries typed without diacritics), "always" or "never"
            grading: Relevance grading of retrieved documents: "none" (the
                generator ignores off-topic passages), "llm" (one batched call)
                or "reranker" (drop scores below min_relevance_score)
            min_relevance_score: Reranker score cut-off for grading="reranker"
            system_prompt: Custom system prompt
            use_cache: Enable semantic caching
            cache_ttl: Cache time-to-live in seconds
            cache_threshold: Similarity threshold for cache hit (0-1)
            use_contextual_chunking: Use Anthropic-style contextual retrieval chunking
            use_hyde: Use HyDE (Hypothetical Document Embeddings) for retrieval
            use_multi_query_rrf: Use multi-query with RRF fusion
            num_query_variations: Number of query variations for multi-query
            use_web_search: Enable web search fallback when retrieval quality is poor
            web_search_provider: Web search provider ("duckduckgo" or "tavily")
            web_search_api_key: API key for web search provider (if needed)
            use_hallucination_check: Enable hallucination verification after generation
            hallucination_threshold: Minimum grounded score to accept answer (0-1)
            use_metadata_enhancement: Enable LLM-based metadata extraction for chunks
        """
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = EmbeddingsManager(
            provider=embedding_provider,
            model_name=embedding_model,
            device=embedding_device,
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
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_parent_context = use_parent_context
        self.parent_fanout = max(1, parent_fanout)
        self._shared_reranker = reranker
        self._shared_reranker_model = reranker_model
        if query_rewrite not in QUERY_REWRITE_MODES:
            raise ValueError(f"query_rewrite must be one of {QUERY_REWRITE_MODES}")
        if grading not in GRADING_MODES:
            raise ValueError(f"grading must be one of {GRADING_MODES}")
        self.query_rewrite = query_rewrite
        self.grading = grading
        self.min_relevance_score = min_relevance_score
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Context window validation
        from src.utils.context_validator import ContextValidator
        self._context_validator = ContextValidator.from_llm_manager(self.llm)

        # Phase 2 options
        self.use_hyde = use_hyde
        self.use_multi_query_rrf = use_multi_query_rrf
        self.num_query_variations = num_query_variations
        self.use_contextual_chunking = use_contextual_chunking

        # Semantic cache
        self._cache = None
        if use_cache:
            from src.utils.cache import SemanticCache
            self._cache = SemanticCache(
                embeddings=self.embeddings,
                threshold=cache_threshold,
                ttl=cache_ttl,
            )
            logger.info(f"Semantic cache enabled (threshold={cache_threshold}, ttl={cache_ttl}s)")

        # Contextual chunker (lazy init)
        self._contextual_chunker = None
        if use_contextual_chunking:
            from src.core.advanced_chunking import ContextualRetrievalChunker
            self._contextual_chunker = ContextualRetrievalChunker(
                llm=self.llm,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            logger.info("Contextual retrieval chunking enabled")

        # Phase 3: Web search fallback
        self._web_searcher = None
        if use_web_search:
            from src.core.web_search import create_web_searcher
            self._web_searcher = create_web_searcher(
                provider=web_search_provider,
                llm=self.llm,
                api_key=web_search_api_key,
            )
            logger.info(f"Web search fallback enabled (provider={web_search_provider})")

        # Phase 3: Hallucination grader
        self._hallucination_grader = None
        self.use_hallucination_check = use_hallucination_check
        if use_hallucination_check:
            from src.agents.hallucination_grader import HallucinationGrader
            self._hallucination_grader = HallucinationGrader(
                llm=self.llm,
                grounded_threshold=hallucination_threshold,
            )
            logger.info(f"Hallucination check enabled (threshold={hallucination_threshold})")

        # Phase 3: Metadata enhancer
        self._metadata_enhancer = None
        if use_metadata_enhancement:
            from src.core.metadata_enhancer import MetadataEnhancer
            self._metadata_enhancer = MetadataEnhancer(llm=self.llm)
            logger.info("Metadata enhancement enabled")

        # Track documents
        self._documents = []
        self._chunks = []

        # Initialize retriever (will be created after documents are added)
        self._retriever = None

        # Queries search while ingestion writes (API background jobs): searches
        # share the index, writes to the store, chunk list and BM25 are exclusive.
        self._index_lock = ReadWriteLock()

        # Persistent stores keep their chunks across restarts: reload them so
        # BM25, counts and parent lookups work without re-indexing.
        if persist_directory or vector_store_provider in ("qdrant", "chroma"):
            self._restore_from_vector_store()

    def _restore_from_vector_store(self) -> None:
        """Load the chunks a persistent vector store already holds."""
        try:
            chunks = self.vector_store.get_all_documents()
        except Exception as e:
            logger.warning(f"Could not load existing chunks from the vector store: {e}")
            return
        if not chunks:
            return

        # One placeholder per source document, so counts and source-root drops work.
        documents = {}
        for chunk in chunks:
            metadata = chunk.metadata or {}
            key = metadata.get("document_id") or metadata.get("source") or metadata.get("file_name")
            if key and key not in documents:
                kept = {
                    name: metadata[name]
                    for name in ("source", "file_name", "document_id", "source_root", "relative_source")
                    if name in metadata
                }
                documents[key] = Document(page_content="", metadata={**kept, "restored": True})

        self._chunks = chunks
        self._documents = list(documents.values())
        self._refresh_retriever()
        logger.info(f"Restored {len(chunks)} chunks from {len(documents)} documents")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt (bilingual Vietnamese/English)."""
        return """You are a helpful AI assistant / Bạn là trợ lý AI hữu ích.
Use the provided context to answer the user's question accurately.
Sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi một cách chính xác.

Rules / Quy tắc:
1. Answer based ONLY on the provided context / Chỉ trả lời dựa trên ngữ cảnh
2. If the context doesn't contain the answer, say "I don't have enough information" / Nếu không đủ thông tin, hãy nói rõ
3. Be concise and accurate / Ngắn gọn và chính xác
4. Answer in the same language as the question / Trả lời bằng ngôn ngữ của câu hỏi

Context / Ngữ cảnh:
{context}

Question / Câu hỏi: {question}"""

    def _get_query_transform_prompt(self) -> str:
        """Get query transformation prompt (bilingual)."""
        return """You are a search query optimizer / Bạn là người tối ưu hóa truy vấn.
Transform the user's question into a better search query.
Chuyển đổi câu hỏi của người dùng thành truy vấn tìm kiếm tốt hơn.

Original question / Câu hỏi gốc: {question}

Transform this into a clearer, more specific search query that will retrieve relevant documents.
Chuyển đổi thành truy vấn rõ ràng hơn, cụ thể hơn.
Return ONLY the transformed query, nothing else."""

    def _get_grading_prompt(self) -> str:
        """Batched document grading prompt (bilingual): one call for all documents."""
        return """You are a document relevance grader / Bạn là người đánh giá tài liệu.
Decide which documents help answer the question.
Xác định những tài liệu nào giúp trả lời câu hỏi.

Question / Câu hỏi: {question}

Documents / Tài liệu:
{documents}

Reply with the IDs of the helpful documents separated by commas (e.g. S1, S3), or NONE.
Trả lời bằng ID các tài liệu hữu ích, cách nhau bởi dấu phẩy (ví dụ S1, S3), hoặc NONE."""

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents to the knowledge base.

        Uses contextual retrieval chunking if enabled (Anthropic pattern),
        otherwise falls back to standard text splitting.

        Args:
            sources: File path(s) or directory path(s)
            metadata: Additional metadata

        Returns:
            Number of chunks added
        """
        # Normalize to list
        if isinstance(sources, (str, Path)):
            sources = [sources]

        # Load documents
        all_docs = []
        for source in sources:
            source = Path(source)
            if source.is_dir():
                docs = self.document_loader.load_directory(source, metadata=metadata)
            else:
                docs = self.document_loader.load(source, metadata=metadata)
            all_docs.extend(docs)

        with self._index_lock.write():
            return self._index_loaded_documents(all_docs)

    def add_files(
        self,
        files: Dict[str, Union[str, Path]],
        source_root: str = "upload",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Index files under stable names; a name indexed again replaces its old chunks.

        For uploads kept in temporary paths: each document is identified by
        ``source_root`` and its name (``relative_source``), not by where the
        file happens to be on disk, so filters, citations and chunk ids stay
        the same when the file is uploaded again.

        Args:
            files: ``{name: path}``, e.g. ``{"tam_cam.pdf": "/tmp/x/tam_cam.pdf"}``
            source_root: Collection the names belong to
            metadata: Extra metadata for every document

        Returns:
            Number of chunks indexed
        """
        loaded = {}
        for name, path in files.items():
            identity = {
                **(metadata or {}),
                "source": f"{source_root}/{name}",
                "source_root": source_root,
                "relative_source": name,
                "file_name": name,
            }
            docs = self.document_loader.load(Path(path), metadata=identity)
            for doc in docs:
                doc.metadata.pop("absolute_source", None)  # a temporary path
            loaded[name] = docs

        with self._index_lock.write():
            for name in loaded:
                self._drop_source_file(source_root, name)
            return self._index_loaded_documents([doc for docs in loaded.values() for doc in docs])

    def refresh_markdown_directory(
        self,
        directory: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        manifest_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Refresh the knowledge base from a Markdown folder.

        The folder is compared against a content-hash manifest. Only what
        changed is touched: chunks of updated and removed files are deleted,
        and added or updated files (plus unchanged files the index does not
        hold, e.g. after a restart on an in-memory store) are embedded again.
        ``force`` rebuilds the whole folder.
        """
        with self._index_lock.write():
            directory = Path(directory).resolve()
            root = str(directory)
            indexer = MarkdownFolderIndexer()
            result, current_manifest = indexer.compare(
                directory,
                Path(manifest_path) if manifest_path else None,
            )

            indexed = {
                (chunk.metadata or {}).get("relative_source")
                for chunk in self._chunks
                if (chunk.metadata or {}).get("source_root") == root
            }
            missing = [name for name in result.unchanged if name not in indexed]
            if not force and not result.changed and not missing:
                return result.to_dict()

            if force or not indexed:
                docs = self.document_loader.load_markdown_directory(
                    directory,
                    metadata=metadata,
                    strict=strict,
                )
                self._drop_source_root(root)
            else:
                for name in result.updated + result.removed:
                    self._drop_source_file(root, name)
                docs = self._load_folder_files(
                    directory, result.added + result.updated + missing, metadata, strict
                )
            chunks_indexed = self._index_loaded_documents(docs)
            indexer.save_manifest(Path(result.manifest_path), current_manifest)

            result.documents_loaded = len(docs)
            result.chunks_indexed = chunks_indexed
            result.rebuilt = True
            return result.to_dict()

    def _load_folder_files(
        self,
        directory: Path,
        names: List[str],
        metadata: Optional[Dict[str, Any]],
        strict: bool,
    ) -> List[Document]:
        """Load some files of a folder with the metadata load_directory would give them."""
        docs = []
        for name in names:
            try:
                loaded = self.document_loader.load(directory / name, metadata=metadata)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Failed to load {directory / name}: {e}")
                continue
            for doc in loaded:
                doc.metadata["source_root"] = str(directory)
                doc.metadata["relative_source"] = name
            docs.extend(loaded)
        return docs

    def _drop_source_file(self, source_root: str, relative_source: str) -> None:
        """Remove one file of a source folder from memory and the vector store."""
        def keep(doc: Document) -> bool:
            metadata = doc.metadata or {}
            return not (
                metadata.get("source_root") == source_root
                and metadata.get("relative_source") == relative_source
            )

        self._documents = [doc for doc in self._documents if keep(doc)]
        self._chunks = [chunk for chunk in self._chunks if keep(chunk)]
        self.vector_store.delete(
            filter={"source_root": source_root, "relative_source": relative_source}
        )
        self.vector_store.persist()

    def _index_loaded_documents(self, all_docs: List[Document]) -> int:
        """Split, index, and register already-loaded documents."""
        if not all_docs:
            self._refresh_retriever()
            return 0

        self._documents.extend(all_docs)

        # Split into chunks (contextual or standard)
        if self._contextual_chunker:
            logger.info("Using contextual retrieval chunking")
            chunks = self._contextual_chunker.split(all_docs)
        else:
            chunks = self.text_splitter.split_documents(all_docs)

        # Enhance metadata if enabled
        if self._metadata_enhancer:
            logger.info(f"Enhancing metadata for {len(chunks)} chunks")
            chunks = self._metadata_enhancer.enhance(chunks)

        # Small child chunks are what gets searched. Each one carries its parent
        # chunk (stable id + text), so use_parent_context can hand the LLM the
        # surrounding passage, and no positional index can go stale.
        child_splitter = TextSplitter(
            chunk_size=max(1, self.text_splitter.chunk_size // 2),
            chunk_overlap=self.text_splitter.chunk_overlap,
        )
        child_chunks = []
        for parent in chunks:
            parent_id = self._parent_id(parent)
            for child in child_splitter.split_documents([parent]):
                child.metadata = {
                    **child.metadata,
                    "parent_id": parent_id,
                    "parent_text": parent.page_content,
                }
                child_chunks.append(child)

        # Use child chunks for retrieval if any were created,
        # otherwise fall back to the parent chunks themselves
        retrieval_chunks = child_chunks if child_chunks else chunks
        MarkdownFolderIndexer().assign_stable_chunk_ids(retrieval_chunks)

        # The store overwrites chunks by id; keep the in-memory list in step.
        new_ids = {chunk.metadata.get("chunk_id") for chunk in retrieval_chunks}
        self._chunks = [
            chunk for chunk in self._chunks
            if (chunk.metadata or {}).get("chunk_id") not in new_ids
        ]
        self._chunks.extend(retrieval_chunks)

        # Add to vector store
        self._add_chunks_to_vector_store(retrieval_chunks)
        self.vector_store.persist()

        # Initialize retriever
        self._refresh_retriever()

        return len(retrieval_chunks)

    def _drop_source_root(self, source_root: str) -> None:
        """Remove one source folder from memory and the vector store."""
        self._documents = [
            doc for doc in self._documents
            if (doc.metadata or {}).get("source_root") != source_root
        ]
        self._chunks = [
            chunk for chunk in self._chunks
            if (chunk.metadata or {}).get("source_root") != source_root
        ]
        self.vector_store.delete(filter={"source_root": source_root})
        self.vector_store.persist()

    @staticmethod
    def _parent_id(parent: Document) -> str:
        """Stable id of a parent chunk: its source, position and text."""
        metadata = parent.metadata or {}
        key = "|".join(
            str(metadata.get(name, ""))
            for name in ("source", "source_sha256", "page", "start_index")
        )
        return hashlib.sha1(f"{key}|{parent.page_content}".encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _expand_to_parents(docs: List[Document]) -> List[Document]:
        """Replace child chunks by their parents, in rank order, without repeats."""
        parents, seen = [], set()
        for doc in docs:
            metadata = doc.metadata or {}
            parent_id, parent_text = metadata.get("parent_id"), metadata.get("parent_text")
            if not parent_id or not parent_text:
                parents.append(doc)
                continue
            if parent_id in seen:
                continue
            seen.add(parent_id)
            kept = {name: value for name, value in metadata.items() if name != "parent_text"}
            parents.append(Document(page_content=parent_text, metadata=kept))
        return parents

    def _add_chunks_to_vector_store(self, chunks: List[Document]) -> None:
        """Add chunks with stable IDs when available."""
        if not chunks:
            return

        ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
        if all(ids):
            self.vector_store.add_documents(chunks, ids=ids)
        else:
            self.vector_store.add_documents(chunks)

    def _refresh_retriever(self) -> None:
        """Refresh hybrid/reranking retriever state from current chunks."""
        if not self._chunks:
            self._retriever = None
            return

        # Cached answers describe the old corpus.
        if self._cache:
            self._cache.clear()

        previous = self._retriever
        self._retriever = RetrieverManager(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            documents=self._chunks,
            k=self.retrieval_k,
            use_hybrid=self.use_hybrid,
            use_reranking=self.use_reranking,
            # Load the cross-encoder once, not after every ingest.
            reranker=getattr(previous, "_reranker", None) or getattr(self, "_shared_reranker", None),
            reranker_model=(
                getattr(previous, "active_reranker_model", None)
                or getattr(self, "_shared_reranker_model", None)
            ),
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add raw texts to the knowledge base.

        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text

        Returns:
            Number of chunks added
        """
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            docs.append(Document(page_content=text, metadata=dict(metadata or {})))

        # Same indexing as files: child chunks with parents, stable ids.
        with self._index_lock.write():
            return self._index_loaded_documents(docs)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: Optional[bool] = None,
        grade_documents: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Answer a question from the knowledge base.

        Pipeline: cache -> query rewrite -> retrieve -> optional grading ->
        "not enough information" or web fallback when nothing relevant was
        found -> generate -> optional hallucination check -> cache.

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Rewrite the query (None: follow ``query_rewrite``)
            grade_documents: Grade relevance (None: follow ``grading``;
                True with grading="none" means one batched LLM call)
            use_reranking: Override reranking for this query (None: as configured)
            filter: Metadata filter, e.g. ``{"file_name": "tam_cam.md"}``; a list
                value matches any of its items

        Returns:
            Answer string
        """
        prepared = self._prepare(question, k, transform_query, grade_documents, use_reranking, filter)
        if prepared.answer is not None:
            return prepared.answer
        return self._finish(prepared, self._generate(prepared, **kwargs))

    def _generate(self, prepared: PreparedQuery, **kwargs) -> str:
        with step("generate", input={"question": prepared.question, "sources": len(prepared.docs)}) as s:
            answer = self.llm.generate(prepared.prompt, **kwargs)
            s.update(output=answer)
        return answer

    def _prepare(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: Optional[bool] = None,
        grade_documents: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> PreparedQuery:
        """Everything before generation; sets ``answer`` when no LLM call is needed."""
        k = k or self.retrieval_k
        # Same question, other filter or k: another answer.
        scope = json.dumps(
            {"k": k, "filter": filter or None, "use_reranking": use_reranking},
            sort_keys=True, ensure_ascii=False, default=str,
        )

        cached = self._cache_get(question, scope)
        if cached is not None:
            return PreparedQuery(question, question, answer=cached, cache_hit=True, cache_scope=scope)

        search_query = self._rewrite_for_search(question, transform_query)
        retrieved = self._retrieve(search_query, k=k, use_reranking=use_reranking, filter=filter)
        docs = self._grade(question, retrieved, grade_documents)
        prepared = PreparedQuery(
            question, search_query, retrieved=retrieved, docs=docs, cache_scope=scope,
        )

        if self._web_searcher and self._is_retrieval_quality_poor(docs, question):
            logger.info("Retrieval quality poor, falling back to web search")
            try:
                with step("web_search", input=search_query) as web_step:
                    web_results = self._web_searcher.search(search_query, num_results=3)
                    prepared.web_docs = self._web_searcher.to_documents(web_results)
                    web_step.update(output={"results": len(prepared.web_docs)})
                logger.info(f"Web search returned {len(prepared.web_docs)} results")
            except Exception as e:
                logger.warning(f"Web search fallback failed: {e}")

        if prepared.web_docs:
            prepared.prompt = self._web_searcher.create_web_answer_prompt(
                question=question,
                web_docs=prepared.web_docs,
                local_docs=docs,
            )
        elif docs:
            prepared.prompt = self.system_prompt.format(
                context=self._build_context(docs),
                question=question,
            )
        else:
            # Nothing relevant: say so instead of letting the LLM guess.
            prepared.answer = NO_CONTEXT_ANSWER
            prepared.abstained = True
        return prepared

    def _finish(self, prepared: PreparedQuery, answer: str, verify: bool = True) -> str:
        """Hallucination check (if enabled) and caching of a generated answer."""
        all_docs = prepared.docs + prepared.web_docs
        if verify and self._hallucination_grader and all_docs:
            context = self._build_context(all_docs)
            with step("verify") as verify_step:
                grade = self._hallucination_grader.grade(answer=answer, context=context)
                verify_step.update(output={"grounded": grade.is_grounded, "score": grade.grounded_score})
            if not grade.is_grounded and grade.grounded_score < self._hallucination_grader.grounded_threshold:
                logger.warning(
                    f"Hallucination detected (score={grade.grounded_score:.2f}), "
                    f"unsupported: {grade.unsupported_claims}"
                )
                # Regenerate with stricter prompt
                answer, _ = self._hallucination_grader.safe_generate(
                    question=prepared.question,
                    context=context,
                    max_retries=1,
                )

        # Web results change quickly: do not cache answers built on them.
        if not prepared.web_docs:
            self._cache_put(prepared.question, answer, prepared.cache_scope)
        return answer

    def _cache_get(self, question: str, scope: Optional[str] = None) -> Optional[str]:
        if not self._cache:
            return None
        try:
            cached = self._cache.get(self.embeddings.embed_query(question), scope=scope)
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
            return None
        if cached is not None:
            logger.debug("Semantic cache hit")
        return cached

    def _cache_put(self, question: str, answer: str, scope: Optional[str] = None) -> None:
        if not self._cache:
            return
        try:
            self._cache.put(self.embeddings.embed_query(question), question, answer, scope=scope)
        except Exception as e:
            logger.debug(f"Cache store failed: {e}")

    def _rewrite_for_search(self, question: str, transform_query: Optional[bool] = None) -> str:
        """LLM query rewrite per ``query_rewrite``; "auto" only for diacritic-free queries.

        Measured on the fairy-tale set, rewriting every query lowered evidence
        recall with reranking (0.941 -> 0.929) but restored queries typed
        without diacritics (0.80 -> 1.00).
        """
        if transform_query is None:
            mode = self.query_rewrite
            transform_query = mode == "always" or (mode == "auto" and not has_diacritics(question))
        if not transform_query:
            return question
        with step("rewrite", input=question) as s:
            rewritten = self._transform_query(question)
            s.update(output=rewritten)
        return rewritten

    def _grade(
        self,
        question: str,
        docs: List[Document],
        grade_documents: Optional[bool] = None,
    ) -> List[Document]:
        """Drop documents that do not help, per ``grading``; may return an empty list."""
        if grade_documents is False or not docs:
            return docs
        mode = self.grading
        if grade_documents and mode == "none":
            mode = "llm"
        if mode == "none" or (mode == "reranker" and self.min_relevance_score is None):
            return docs
        with step("grade", input={"mode": mode, "documents": len(docs)}) as s:
            if mode == "llm":
                kept = self._grade_documents(question, docs)
            else:
                kept = [
                    doc for doc in docs
                    if (doc.metadata or {}).get("relevance_score", self.min_relevance_score)
                    >= self.min_relevance_score
                ]
            s.update(output={"kept": len(kept)})
        return kept

    def _is_retrieval_quality_poor(self, docs: List[Document], question: str) -> bool:
        """
        Check if retrieval quality is poor enough to warrant web search fallback.

        Returns True if:
        - No documents retrieved, OR
        - All documents were graded as irrelevant (empty after grading)
        """
        if not docs:
            return True

        # Check if docs have relevance metadata from grading
        # The _grade_documents method already filters, so if we get here
        # with docs, at least some were deemed relevant
        return False

    def _retrieve(
        self,
        query: str,
        k: int = 5,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Internal retrieval method supporting multiple strategies.

        Checks in order:
        1. HyDE (if enabled)
        2. Multi-query RRF (if enabled)
        3. Standard search
        """
        with step("retrieve", as_type="retriever", input={"query": query, "k": k, "filter": filter}) as s:
            with self._index_lock.read():
                docs = self._retrieve_unlocked(query, k, use_reranking, filter)
            s.update(output=_trace_sources(docs))
        return docs

    def _retrieve_unlocked(
        self,
        query: str,
        k: int,
        use_reranking: Optional[bool],
        filter: Optional[Dict[str, Any]],
    ) -> List[Document]:
        if self._retriever is None:
            return self.vector_store.similarity_search(query, k=k, filter=filter)

        # HyDE: generate hypothetical answer, search with that
        if self.use_hyde:
            docs = self._retriever.hyde_search(query, k=k, llm=self.llm, filter=filter)
            if docs:
                return self._expand_to_parents(docs)[:k] if self.use_parent_context else docs

        # Multi-query RRF: generate variations, fuse with RRF
        if self.use_multi_query_rrf:
            docs = self._retriever.multi_query_rrf_search(
                query, k=k, llm=self.llm,
                num_queries=self.num_query_variations,
                filter=filter,
            )
            if docs:
                return self._expand_to_parents(docs)[:k] if self.use_parent_context else docs

        # Standard search (hybrid + reranking)
        return self._search(query, k=k, use_reranking=use_reranking, filter=filter)

    def _search(
        self,
        query: str,
        k: int,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Hybrid/rerank search; with parent context, children are searched and parents returned."""
        if not self.use_parent_context:
            return self._retriever.search(
                query, k=k, use_hybrid=use_hybrid, use_reranking=use_reranking, filter=filter
            )

        reranking = self._retriever.config.use_reranking if use_reranking is None else use_reranking
        children = self._retriever.search(
            query, k=k * self.parent_fanout, use_hybrid=use_hybrid, use_reranking=False,
            filter=filter,
        )
        parents = self._expand_to_parents(children)
        if reranking:
            # Rerank the passages the LLM will actually read.
            return self._retriever.rerank(query, parents, k=k)
        return parents[:k]

    def query_detailed(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: Optional[bool] = None,
        grade_documents: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer a question and report how: same pipeline as query().

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Rewrite the query (None: follow ``query_rewrite``)
            grade_documents: Grade relevance (None: follow ``grading``)
            use_reranking: Override reranking for this query (None: as configured)
            filter: Metadata filter, e.g. ``{"file_name": "tam_cam.md"}``; a list
                value matches any of its items

        Returns:
            Dict with the answer, the search query, ``relevant_docs`` (every
            source given to the LLM), ``citations`` (the sources the answer
            actually cites; all of them when it cites none) and ``steps``
            (each stage with its duration in ms, LLM calls with token usage)
        """
        with record_steps() as steps:
            prepared = self._prepare(question, k, transform_query, grade_documents, use_reranking, filter)
            if prepared.answer is None:
                answer = self._finish(prepared, self._generate(prepared, **kwargs))
            else:
                answer = prepared.answer
        result = self._detailed_result(prepared, answer)
        result["steps"] = steps
        return result

    def _detailed_result(self, prepared: PreparedQuery, answer: str) -> Dict[str, Any]:
        sources = self.sources_for(prepared)
        cited, invalid = self.cited_sources(answer, sources)
        return {
            "answer": answer,
            "original_query": prepared.question,
            "transformed_query": None if prepared.cache_hit else prepared.search_query,
            "relevant_docs": sources,
            "citations": cited,
            "invalid_citations": invalid,
            "total_docs_retrieved": len(prepared.retrieved),
            "relevant_docs_count": len(prepared.docs),
            "abstained": prepared.abstained,
            "cache_hit": prepared.cache_hit,
        }

    def sources_for(self, prepared: PreparedQuery) -> List[Dict[str, Any]]:
        """The sources in a prepared query's prompt, labelled [S1], [S2]... as there."""
        return self._format_sources(prepared.web_docs or prepared.docs)

    @staticmethod
    def cited_sources(answer: str, sources: List[Dict[str, Any]]):
        """Sources the answer cites as [S#], and cited ids that do not exist.

        An answer without any [S#] marker keeps every source, as before.
        """
        cited_ids = re.findall(r"\[(S\d+)\]", answer or "")
        if not cited_ids:
            return sources, []
        known = {source["source_id"] for source in sources}
        cited = [source for source in sources if source["source_id"] in set(cited_ids)]
        invalid = sorted(set(cited_ids) - known, key=lambda sid: int(sid[1:]))
        return cited, invalid

    def query_multimodal(
        self,
        question: str,
        media: List[Dict[str, str]],
        k: Optional[int] = None,
        use_retrieval: bool = True,
        llm: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Analyze images/videos and enrich the answer with advanced retrieval."""
        k = k or self.retrieval_k
        docs = self._retrieve(question, k=k) if use_retrieval and self.num_chunks else []
        context = self._build_context(docs) if docs else ""
        prompt = build_multimodal_prompt(question, context)
        model_client = llm or self.llm
        answer = model_client.generate_multimodal(prompt, media, **kwargs)
        sources = self._format_sources(docs)

        return {
            "answer": answer,
            "sources": sources,
            "citations": sources,
            "media_count": len(media),
            "model": model_client.config.model,
        }

    def _transform_query(self, question: str) -> str:
        """Transform query for better retrieval."""
        try:
            prompt = self._get_query_transform_prompt().format(question=question)
            transformed = self.llm.generate(prompt)
            return transformed.strip()
        except Exception as e:
            logger.warning(f"Query transformation failed, using original: {e}")
            return question

    def _grade_documents(
        self,
        question: str,
        docs: List[Document]
    ) -> List[Document]:
        """Grade all documents for relevance in one LLM call; may return an empty list."""
        listing = "\n\n".join(
            f"[S{i}] {doc.page_content[:500]}" for i, doc in enumerate(docs, 1)
        )
        try:
            reply = self.llm.generate(
                self._get_grading_prompt().format(question=question, documents=listing)
            )
        except Exception as e:
            logger.warning(f"Document grading failed, keeping all documents: {e}")
            return docs

        ids = {int(n) for n in re.findall(r"S\s*(\d+)", reply.upper())}
        if not ids:
            if "NONE" in reply.upper() or "KHÔNG" in reply.upper():
                return []
            logger.warning(f"Unreadable grading reply, keeping all documents: {reply[:80]!r}")
            return docs
        return [doc for i, doc in enumerate(docs, 1) if i in ids]

    def _build_context(self, docs: List[Document]) -> str:
        """Build context from documents with context window validation."""
        context_parts = [
            "Source IDs are shown as [S1], [S2], etc. Cite them when using facts."
        ]
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("url")
                or f"Document {i}"
            )
            context_parts.append(f"[S{i}] Source: {source}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Validate context fits within model's window
        if self._context_validator:
            result = self._context_validator.validate(
                prompt=context,
                system_prompt=self.system_prompt,
            )

            if result.warning:
                logger.warning(result.warning)

            if result.is_too_large and result.truncated_prompt:
                logger.warning(
                    f"Context truncated: {result.prompt_tokens:,} → "
                    f"{self._context_validator.count_tokens(result.truncated_prompt):,} tokens"
                )
                return result.truncated_prompt

        return context

    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Format retrieved documents with stable source IDs for citations."""
        sources = []

        for i, doc in enumerate(docs, 1):
            # parent_text duplicates a passage the client does not need.
            metadata = {
                name: value for name, value in (doc.metadata or {}).items()
                if name != "parent_text"
            }
            content = doc.page_content
            source = (
                metadata.get("source")
                or metadata.get("file_name")
                or metadata.get("url")
                or f"Document {i}"
            )

            sources.append({
                "source_id": f"S{i}",
                "source": source,
                "content": content[:300] + "..." if len(content) > 300 else content,
                "metadata": metadata,
            })

        return sources

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve documents with specific strategy.

        Args:
            query: Search query
            k: Number of results
            use_hybrid: Override hybrid setting
            use_reranking: Override reranking setting
            filter: Metadata filter (equality; a list value matches any item)

        Returns:
            List of relevant documents
        """
        k = k or self.retrieval_k

        with step("retrieve", as_type="retriever", input={"query": query, "k": k, "filter": filter}) as s:
            with self._index_lock.read():
                if self._retriever is None:
                    docs = self.vector_store.similarity_search(query, k=k, filter=filter)
                else:
                    docs = self._search(
                        query, k=k, use_hybrid=use_hybrid, use_reranking=use_reranking, filter=filter
                    )
            s.update(output=_trace_sources(docs))
        return docs

    @property
    def num_documents(self) -> int:
        """Number of loaded documents."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

    @property
    def context_info(self) -> Dict[str, Any]:
        """
        Get context window information.

        Returns:
            Dict with context window size, available tokens, and current usage
        """
        if not self._context_validator:
            return {"error": "Context validator not initialized"}

        # Estimate current context size
        total_chunk_chars = sum(len(c.page_content) for c in self._chunks)
        estimated_context_tokens = self._context_validator.estimate_tokens(
            "\n\n".join(c.page_content for c in self._chunks[:self.retrieval_k])
        )

        return {
            "context_window": self._context_validator.context_window,
            "available_tokens": self._context_validator.available_tokens,
            "max_output_tokens": self._context_validator.reserve_tokens,
            "retrieval_k": self.retrieval_k,
            "chunk_size_avg": total_chunk_chars // max(len(self._chunks), 1),
            "estimated_context_tokens": estimated_context_tokens,
            "usage_ratio": round(
                estimated_context_tokens / self._context_validator.available_tokens, 3
            ) if self._context_validator.available_tokens > 0 else 1.0,
            "model": self.llm.config.model,
            "provider": self.llm.config.provider,
        }

    def stream(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: Optional[bool] = None,
        grade_documents: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response tokens.

        Same pipeline as query(); a cached or "not enough information" answer
        is yielded whole. Used by ConversationalRAG.stream().

        Args:
            question: Question to ask
            k: Number of documents
            transform_query: Rewrite the query (None: follow ``query_rewrite``)
            grade_documents: Grade relevance (None: follow ``grading``)
            use_reranking: Override reranking for this query (None: as configured)
            filter: Metadata filter, e.g. ``{"file_name": "tam_cam.md"}``; a list
                value matches any of its items

        Yields:
            Response tokens
        """
        prepared = self._prepare(question, k, transform_query, grade_documents, use_reranking, filter)
        yield from self.stream_prepared(prepared, **kwargs)

    def prepare(
        self,
        question: str,
        k: Optional[int] = None,
        transform_query: Optional[bool] = None,
        grade_documents: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> PreparedQuery:
        """
        First half of the pipeline, for callers that generate separately.

        Cache lookup, query rewrite, retrieval, grading and the prompt; then
        ``stream_prepared()`` generates. An async server runs the two halves in
        worker threads and can send ``sources_for(prepared)`` before the first
        token: they are exactly the passages of the prompt. Arguments as in
        ``query()``.
        """
        return self._prepare(question, k, transform_query, grade_documents, use_reranking, filter)

    def stream_prepared(self, prepared: PreparedQuery, **kwargs) -> Generator[str, None, None]:
        """Second half: stream the answer to a ``prepare()`` result, then cache it."""
        if prepared.answer is not None:
            yield prepared.answer
            return

        full_response = []
        for token in self.llm.stream(prepared.prompt, **kwargs):
            full_response.append(token)
            yield token

        # Tokens are already sent: no hallucination rewrite, only caching.
        self._finish(prepared, "".join(full_response), verify=False)

    @property
    def cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if cache is enabled."""
        if self._cache:
            return self._cache.stats()
        return None
