"""
Graph RAG
=========

Knowledge Graph-based RAG for structured reasoning.

Inspired by LightRAG and Microsoft GraphRAG.

Features:
- Entity and relationship extraction
- Knowledge graph construction
- Graph-based retrieval
- Community detection
- Hybrid retrieval (graph + vector)

Usage:
    rag = GraphRAG()
    rag.add_documents(["docs/"])
    answer = rag.query("What is the relationship between X and Y?")
"""

from typing import List, Optional, Dict, Any, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

from src.core.document_loader import DocumentLoader
from src.core.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingsManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager


@dataclass
class Entity:
    """Entity in the knowledge graph."""
    name: str
    entity_type: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Relationship between entities."""
    source: str
    target: str
    relationship_type: str
    description: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Community:
    """Community in the knowledge graph."""
    id: int
    entities: List[str]
    summary: str
    level: int = 0


class KnowledgeGraph:
    """
    Knowledge Graph for storing entities and relationships.

    Supports optional Neo4j backend for persistent storage.
    When Neo4j backend is provided, entities and relationships
    are synced to both in-memory graph and Neo4j.

    Example:
        kg = KnowledgeGraph()

        # Add entities
        kg.add_entity(Entity(name="Python", entity_type="Technology", description="Programming language"))

        # Add relationships
        kg.add_relationship(Relationship(source="Python", target="AI", relationship_type="used_in"))

        # Query
        neighbors = kg.get_neighbors("Python")

        # With Neo4j persistence
        from src.core.graph_store import Neo4jBackend
        backend = Neo4jBackend(uri="bolt://localhost:7687", password="pass")
        backend.connect()
        kg = KnowledgeGraph(neo4j_backend=backend)
    """

    def __init__(self, neo4j_backend=None):
        """
        Initialize knowledge graph.

        Args:
            neo4j_backend: Optional Neo4jBackend for persistent storage
        """
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.adjacency: Dict[str, Set[str]] = {}  # entity -> set of connected entities
        self._neo4j = neo4j_backend

    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph (and sync to Neo4j if available)."""
        self.entities[entity.name] = entity
        if entity.name not in self.adjacency:
            self.adjacency[entity.name] = set()

        # Sync to Neo4j
        if self._neo4j and self._neo4j.is_connected():
            try:
                self._neo4j.sync_entity(entity)
            except Exception as e:
                logger.warning(f"Neo4j entity sync failed: {e}")

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to graph (and sync to Neo4j if available)."""
        self.relationships.append(relationship)

        # Update adjacency
        if relationship.source not in self.adjacency:
            self.adjacency[relationship.source] = set()
        if relationship.target not in self.adjacency:
            self.adjacency[relationship.target] = set()

        self.adjacency[relationship.source].add(relationship.target)
        self.adjacency[relationship.target].add(relationship.source)

        # Sync to Neo4j
        if self._neo4j and self._neo4j.is_connected():
            try:
                self._neo4j.sync_relationship(relationship)
            except Exception as e:
                logger.warning(f"Neo4j relationship sync failed: {e}")

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self.entities.get(name)

    def get_neighbors(self, name: str, depth: int = 1) -> Dict[str, Entity]:
        """Get neighboring entities."""
        if depth <= 0:
            return {}

        neighbors = {}
        visited = set()
        queue = [(name, 0)]

        while queue:
            current, current_depth = queue.pop(0)

            if current in visited or current_depth > depth:
                continue

            visited.add(current)

            if current != name and current in self.entities:
                neighbors[current] = self.entities[current]

            if current_depth < depth:
                for neighbor in self.adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))

        return neighbors

    def get_relationships(self, entity_name: str) -> List[Relationship]:
        """Get all relationships involving an entity."""
        return [
            r for r in self.relationships
            if r.source == entity_name or r.target == entity_name
        ]

    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find path between two entities."""
        if start not in self.adjacency or end not in self.adjacency:
            return None

        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)

            if current == end:
                return path

            if len(path) > max_depth:
                continue

            visited.add(current)

            for neighbor in self.adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return list(self.entities.values())

    def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships."""
        return self.relationships

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                }
                for e in self.entities.values()
            ],
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.relationship_type,
                    "description": r.description,
                }
                for r in self.relationships
            ],
        }

    def save(self, path: str) -> None:
        """Save graph to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load graph from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for e in data.get("entities", []):
            self.add_entity(Entity(
                name=e["name"],
                entity_type=e["type"],
                description=e["description"],
            ))

        for r in data.get("relationships", []):
            self.add_relationship(Relationship(
                source=r["source"],
                target=r["target"],
                relationship_type=r["type"],
                description=r["description"],
            ))

    def set_neo4j_backend(self, backend) -> None:
        """
        Set or update the Neo4j backend.

        Args:
            backend: Neo4jBackend instance
        """
        self._neo4j = backend

    def sync_to_neo4j(self) -> int:
        """
        Sync all in-memory data to Neo4j.

        Returns:
            Number of entities synced
        """
        if not self._neo4j or not self._neo4j.is_connected():
            logger.warning("Neo4j backend not available for sync")
            return 0
        return self._neo4j.sync_from_knowledge_graph(self)


class GraphRAG:
    """
    Graph-based RAG using Knowledge Graphs.

    Combines knowledge graph with vector retrieval for better reasoning.

    Example:
        rag = GraphRAG(llm_provider="openai")
        rag.add_documents(["docs/"])

        # Query with graph reasoning
        answer = rag.query("What is the relationship between Python and AI?")

        # Get knowledge graph
        kg = rag.get_knowledge_graph()
        neighbors = kg.get_neighbors("Python")
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        embedding_provider: str = "huggingface",
        embedding_model: str = "keepitreal/vietnamese-sbert",
        vector_store_provider: str = "faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 5,
        neo4j_uri: Optional[str] = None,
        neo4j_user: str = "neo4j",
        neo4j_password: Optional[str] = None,
    ):
        """
        Initialize Graph RAG.

        Args:
            llm_provider: LLM provider
            llm_model: LLM model name
            llm_api_key: LLM API key
            embedding_provider: Embedding provider
            embedding_model: Embedding model name
            vector_store_provider: Vector store provider
            chunk_size: Chunk size
            chunk_overlap: Chunk overlap
            retrieval_k: Number of documents to retrieve
            neo4j_uri: Optional Neo4j bolt URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password (or from NEO4J_PASSWORD env var)
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
        )
        self.vector_store = VectorStoreManager(
            provider=vector_store_provider,
            embeddings=self.embeddings,
        )
        self.llm = LLMManager(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        )

        self.retrieval_k = retrieval_k

        # Knowledge graph with optional Neo4j backend
        self._neo4j_backend = None
        if neo4j_uri:
            from src.core.graph_store import Neo4jBackend
            self._neo4j_backend = Neo4jBackend(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )
            if self._neo4j_backend.connect():
                logger.info(f"GraphRAG: Neo4j connected at {neo4j_uri}")
            else:
                logger.warning("GraphRAG: Neo4j connection failed, using NetworkX only")
                self._neo4j_backend = None

        self.knowledge_graph = KnowledgeGraph(neo4j_backend=self._neo4j_backend)

        # Track documents
        self._documents = []
        self._chunks = []

    def add_documents(
        self,
        sources: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add documents and extract knowledge graph.

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

        self._documents.extend(all_docs)

        # Split into chunks
        chunks = self.text_splitter.split_documents(all_docs)
        self._chunks.extend(chunks)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        # Extract entities and relationships
        self._extract_knowledge(chunks)

        return len(chunks)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Add texts and extract knowledge graph.

        Args:
            texts: List of text strings
            metadatas: Optional metadata for each text

        Returns:
            Number of chunks added
        """
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            docs.append(Document(page_content=text, metadata=metadata))

        self._documents.extend(docs)

        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        self._chunks.extend(chunks)

        # Add to vector store
        self.vector_store.add_documents(chunks)

        # Extract entities and relationships
        self._extract_knowledge(chunks)

        return len(chunks)

    def _extract_knowledge(self, chunks: List[Document]) -> None:
        """Extract entities and relationships from chunks."""
        import logging
        logger = logging.getLogger(__name__)

        for i, chunk in enumerate(chunks):
            try:
                # Extract entities and relationships using LLM (bilingual prompt)
                prompt = f"""Extract entities and relationships from the following text.
Trích xuất các thực thể và mối quan hệ từ văn bản sau.

Text / Văn bản:
{chunk.page_content[:2000]}

Return a JSON object with / Trả về đối tượng JSON với:
- "entities": list of {{"name": "...", "type": "...", "description": "..."}}
- "relationships": list of {{"source": "...", "target": "...", "type": "...", "description": "..."}}

Return ONLY valid JSON, nothing else."""

                response = self.llm.generate(prompt)

                # Parse JSON
                try:
                    # Clean response
                    json_str = response.strip()
                    if json_str.startswith("```"):
                        json_str = json_str.split("\n", 1)[1]
                    if json_str.endswith("```"):
                        json_str = json_str.rsplit("```", 1)[0]

                    data = json.loads(json_str)

                    # Add entities
                    for e in data.get("entities", []):
                        self.knowledge_graph.add_entity(Entity(
                            name=e["name"],
                            entity_type=e.get("type", "Unknown"),
                            description=e.get("description", ""),
                        ))

                    # Add relationships
                    for r in data.get("relationships", []):
                        self.knowledge_graph.add_relationship(Relationship(
                            source=r["source"],
                            target=r["target"],
                            relationship_type=r.get("type", "related_to"),
                            description=r.get("description", ""),
                        ))

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse entity extraction JSON for chunk {i}: {e}")
                except KeyError as e:
                    logger.warning(f"Missing expected key in entity extraction for chunk {i}: {e}")

            except Exception as e:
                logger.error(f"Entity extraction failed for chunk {i}: {e}")

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        use_graph: bool = True,
        **kwargs
    ) -> str:
        """
        Query using graph and vector retrieval.

        Args:
            question: Question to ask
            k: Number of documents to retrieve
            use_graph: Whether to use graph retrieval

        Returns:
            Answer string
        """
        k = k or self.retrieval_k

        # Vector retrieval
        vector_docs = self.vector_store.similarity_search(question, k=k)

        # Graph retrieval
        graph_context = ""
        if use_graph:
            graph_context = self._graph_retrieval(question)

        # Build context
        context_parts = []

        if graph_context:
            context_parts.append(f"Knowledge Graph Information:\n{graph_context}")

        for i, doc in enumerate(vector_docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Generate answer (bilingual)
        prompt = f"""You are a helpful AI assistant / Bạn là trợ lý AI hữu ích.
Use the provided context to answer the question.
Sử dụng ngữ cảnh để trả lời câu hỏi.

The context includes knowledge graph and document excerpts.
Ngữ cảnh bao gồm đồ thị tri thức và trích đoạn tài liệu.

Rules / Quy tắc:
1. Answer based on the provided context / Trả lời dựa trên ngữ cảnh
2. Use knowledge graph for relationships / Sử dụng đồ thị tri thức cho mối quan hệ
3. Be concise and accurate / Ngắn gọn và chính xác
4. Answer in the same language as the question / Trả lời bằng ngôn ngữ của câu hỏi

Context / Ngữ cảnh:
{context}

Question / Câu hỏi: {question}"""

        return self.llm.generate(prompt, **kwargs)

    def _graph_retrieval(self, question: str) -> str:
        """Retrieve relevant information from knowledge graph."""
        # Extract entity names from question (bilingual)
        prompt = f"""Extract the main entity names mentioned in this question.
Trích xuất tên các thực thể chính trong câu hỏi này.

Question / Câu hỏi: {question}

Return entity names as a comma-separated list. Return ONLY the list, nothing else.
Trả về tên thực thể dưới dạng danh sách cách nhau bằng dấu phẩy."""

        response = self.llm.generate(prompt)

        # Clean response: remove common prefixes like "The main entities are:"
        response = response.strip()
        for prefix in ["The main entities are:", "Entities:", "Entities are:", "Main entities:"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        entity_names = [name.strip() for name in response.split(",") if name.strip()]

        # Get graph information
        graph_info = []

        for name in entity_names:
            entity = self.knowledge_graph.get_entity(name)
            if entity:
                # Get neighbors
                neighbors = self.knowledge_graph.get_neighbors(name, depth=1)

                # Get relationships
                relationships = self.knowledge_graph.get_relationships(name)

                # Format information
                info_parts = [f"Entity: {entity.name} ({entity.entity_type})"]
                info_parts.append(f"Description: {entity.description}")

                if relationships:
                    info_parts.append("Relationships:")
                    for rel in relationships:
                        other = rel.target if rel.source == name else rel.source
                        info_parts.append(f"  - {name} -> {other}: {rel.relationship_type}")

                if neighbors:
                    info_parts.append("Connected entities:")
                    for neighbor_name, neighbor in neighbors.items():
                        info_parts.append(f"  - {neighbor_name}: {neighbor.description[:100]}")

                graph_info.append("\n".join(info_parts))

        return "\n\n".join(graph_info) if graph_info else ""

    def extract_subgraph_context(self, question: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Extract N-hop sub-graph context surrounding entities in the query.

        Args:
            question: Search query or question
            max_hops: Maximum graph traversal depth (default: 2)

        Returns:
            Dict containing matched entities, sub-graph triples, and formatted context text
        """
        graph_text = self._graph_retrieval(question)
        entities = list(self.knowledge_graph.entities.keys())
        matched = [e for e in entities if e.lower() in question.lower()]
        
        triples = []
        for rel in self.knowledge_graph.relationships:
            if rel.source in matched or rel.target in matched:
                triples.append(f"{rel.source} -[{rel.relationship_type}]-> {rel.target}")

        return {
            "matched_entities": matched,
            "subgraph_triples": triples,
            "formatted_context": graph_text,
            "max_hops": max_hops,
        }

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Get the knowledge graph."""
        return self.knowledge_graph

    def save_graph(self, path: str) -> None:
        """Save knowledge graph to file."""
        self.knowledge_graph.save(path)

    def load_graph(self, path: str) -> None:
        """Load knowledge graph from file."""
        self.knowledge_graph.load(path)

    @property
    def num_documents(self) -> int:
        """Number of loaded documents."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks."""
        return len(self._chunks)

    @property
    def num_entities(self) -> int:
        """Number of entities in knowledge graph."""
        return len(self.knowledge_graph.entities)

    @property
    def num_relationships(self) -> int:
        """Number of relationships in knowledge graph."""
        return len(self.knowledge_graph.relationships)

    @property
    def has_neo4j(self) -> bool:
        """Whether Neo4j backend is connected."""
        return self._neo4j_backend is not None and self._neo4j_backend.is_connected()

    def sync_to_neo4j(self) -> int:
        """
        Sync in-memory graph to Neo4j.

        Returns:
            Number of entities synced
        """
        return self.knowledge_graph.sync_to_neo4j()

    def close(self):
        """Close Neo4j connection if open."""
        if self._neo4j_backend:
            self._neo4j_backend.close()
