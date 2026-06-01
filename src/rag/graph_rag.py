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

from typing import List, Optional, Dict, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import json

from langchain_core.documents import Document

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

    Example:
        kg = KnowledgeGraph()

        # Add entities
        kg.add_entity(Entity(name="Python", entity_type="Technology", description="Programming language"))

        # Add relationships
        kg.add_relationship(Relationship(source="Python", target="AI", relationship_type="used_in"))

        # Query
        neighbors = kg.get_neighbors("Python")
    """

    def __init__(self):
        """Initialize knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.adjacency: Dict[str, Set[str]] = {}  # entity -> set of connected entities

    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph."""
        self.entities[entity.name] = entity
        if entity.name not in self.adjacency:
            self.adjacency[entity.name] = set()

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to graph."""
        self.relationships.append(relationship)

        # Update adjacency
        if relationship.source not in self.adjacency:
            self.adjacency[relationship.source] = set()
        if relationship.target not in self.adjacency:
            self.adjacency[relationship.target] = set()

        self.adjacency[relationship.source].add(relationship.target)
        self.adjacency[relationship.target].add(relationship.source)

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
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_provider: str = "faiss",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        retrieval_k: int = 5,
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

        # Knowledge graph
        self.knowledge_graph = KnowledgeGraph()

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
        for chunk in chunks:
            try:
                # Extract entities and relationships using LLM
                prompt = f"""Extract entities and relationships from the following text.

Text:
{chunk.page_content[:2000]}

Return a JSON object with:
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

                except (json.JSONDecodeError, KeyError):
                    # Skip if parsing fails
                    pass

            except Exception:
                # Skip if extraction fails
                pass

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

        # Generate answer
        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the question.

The context includes both knowledge graph information and document excerpts.

Rules:
1. Answer based on the provided context
2. Use knowledge graph information for relationships and connections
3. Be concise and accurate

Context:
{context}

Question: {question}"""

        return self.llm.generate(prompt, **kwargs)

    def _graph_retrieval(self, question: str) -> str:
        """Retrieve relevant information from knowledge graph."""
        # Extract entity names from question
        prompt = f"""Extract the main entity names mentioned in this question.

Question: {question}

Return entity names as a comma-separated list. Return ONLY the list, nothing else."""

        response = self.llm.generate(prompt)
        entity_names = [name.strip() for name in response.split(",")]

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
