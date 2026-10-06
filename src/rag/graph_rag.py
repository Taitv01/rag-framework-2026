"""
Graph RAG
=========

Knowledge graph RAG in the spirit of Microsoft GraphRAG: AdvancedRAG with its
knowledge graph index on.

Features:
- Entity and relationship extraction (one LLM call per document, or per
  ~4000 characters of a long one)
- Entities merged across documents, Louvain communities, one LLM report each
- Graph stored beside the chunks (survives restarts), filters apply to it
- Entity profiles and community reports join the context when they outrank
  passages; no LLM call at query time
- KnowledgeGraph view for traversal, export and optional Neo4j sync

Usage:
    rag = GraphRAG()
    rag.add_documents(["docs/"])
    answer = rag.query("What is the relationship between X and Y?")
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.rag.advanced_rag import AdvancedRAG
from src.rag.document_cards import document_key

logger = logging.getLogger(__name__)


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


class GraphRAG(AdvancedRAG):
    """
    AdvancedRAG with its knowledge graph on (``use_graph=True``).

    Documents are indexed as passages (hybrid search, reranking, parent
    context, document cards) and as a knowledge graph: entities and
    relationships extracted by the LLM, merged across documents, grouped into
    Louvain communities with one report each (``src/rag/graph_index.py``).
    The graph is stored beside the chunks and survives restarts.

    Questions go through AdvancedRAG's single pipeline, usually one LLM call:
    graph entities and community reports join the context when they outrank
    passages. ``knowledge_graph`` is a KnowledgeGraph view of the merged
    graph (traversal, export, Neo4j), refreshed after every change.

    Example:
        rag = GraphRAG(llm_provider="anthropic", persist_directory="./data")
        rag.add_documents(["stories/"])
        answer = rag.query("Những truyện nào có thủy cung?")
        neighbors = rag.get_knowledge_graph().get_neighbors("Thạch Sanh")
    """

    def __init__(
        self,
        *args,
        neo4j_uri: Optional[str] = None,
        neo4j_user: str = "neo4j",
        neo4j_password: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            *args, **kwargs: AdvancedRAG options (``use_graph`` defaults to True)
            neo4j_uri: Optional Neo4j bolt URI (e.g., "bolt://localhost:7687"):
                the graph is also written there after every change
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password (or from NEO4J_PASSWORD env var)
        """
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
        kwargs.setdefault("use_graph", True)
        super().__init__(*args, **kwargs)

    def _graph_changed(self) -> None:
        """Rebuild the KnowledgeGraph view (and Neo4j copy) from the graph index."""
        graph = getattr(self, "_graph", None)
        if graph is None:
            return
        view = graph.view
        knowledge_graph = KnowledgeGraph(neo4j_backend=self._neo4j_backend)
        for entity in view.entities.values():
            knowledge_graph.add_entity(Entity(
                name=entity["name"],
                entity_type=entity["type"],
                description=" ".join(text for text in entity["descriptions"].values() if text),
                metadata={"documents": view._document_names(sorted(entity["descriptions"]))},
            ))
        directed: Dict[tuple, List[str]] = {}
        for document in view.documents.values():
            for pair, description in sorted(document.relationships.items()):
                directed.setdefault(pair, []).append(description)
        for (source, target), descriptions in directed.items():
            knowledge_graph.add_relationship(Relationship(
                source=view.entities[source]["name"],
                target=view.entities[target]["name"],
                relationship_type="related_to",
                description=" | ".join(text for text in descriptions if text),
                weight=float(len(descriptions)),
            ))
        self.knowledge_graph = knowledge_graph

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Add raw texts; a text without a source becomes its own document
        (``document_id`` "text/<hash>"), so it gets a graph and a card.
        """
        metadatas = [dict(metadata or {}) for metadata in (metadatas or [{}] * len(texts))]
        for text, metadata in zip(texts, metadatas):
            if not document_key(metadata):
                metadata["document_id"] = "text/" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return super().add_texts(texts, metadatas)

    def get_communities(self) -> List[Community]:
        """Communities that have a report, largest first (``summary`` is the report)."""
        graph = getattr(self, "_graph", None)
        if graph is None:
            return []
        communities = sorted(
            (community for community in graph.communities if community.id in graph.reports),
            key=lambda community: (-len(community.entities), community.id),
        )
        return [
            Community(
                id=index,
                entities=[graph.view.entities[key]["name"] for key in community.entities],
                summary=graph.reports[community.id].page_content,
            )
            for index, community in enumerate(communities)
        ]

    def _extract_entity_names(self, question: str) -> List[str]:
        """Extract candidate entity names from a question using the configured LLM."""
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

        return [name.strip() for name in response.split(",") if name.strip()]

    def extract_subgraph_context(self, question: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Extract N-hop sub-graph context surrounding entities in the query.

        Args:
            question: Search query or question
            max_hops: Maximum graph traversal depth (default: 2)

        Returns:
            Dict containing matched entities, sub-graph triples, and formatted context text
        """
        if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 0:
            raise ValueError("max_hops must be a non-negative integer")

        entities = list(self.knowledge_graph.entities.keys())
        folded_question = question.casefold()
        matched = [name for name in entities if name.casefold() in folded_question]

        # If a question uses an alias or an indirect reference, reuse the LLM
        # entity extractor and map its output back to canonical graph names.
        if not matched:
            extracted = self._extract_entity_names(question)
            canonical_names = {name.casefold(): name for name in entities}
            matched = [
                canonical_names[name.casefold()]
                for name in extracted
                if name.casefold() in canonical_names
            ]

        reached = set(matched)
        frontier = set(matched)
        selected_relationships: List[Relationship] = []
        selected_ids: Set[int] = set()

        # Traverse the in-memory graph as undirected, matching KnowledgeGraph's
        # adjacency semantics. Each iteration adds exactly one hop.
        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for index, relationship in enumerate(self.knowledge_graph.relationships):
                if relationship.source in frontier:
                    next_frontier.add(relationship.target)
                elif relationship.target in frontier:
                    next_frontier.add(relationship.source)
                else:
                    continue

                if index not in selected_ids:
                    selected_relationships.append(relationship)
                    selected_ids.add(index)

            next_frontier -= reached
            if not next_frontier:
                break
            reached.update(next_frontier)
            frontier = next_frontier

        triples = [
            f"{rel.source} -[{rel.relationship_type}]-> {rel.target}"
            for rel in selected_relationships
        ]

        context_parts = []
        for name in sorted(reached):
            entity = self.knowledge_graph.entities.get(name)
            if entity:
                context_parts.append(
                    f"Entity: {entity.name} ({entity.entity_type})\n"
                    f"Description: {entity.description}"
                )
        if triples:
            context_parts.append("Relationships:\n" + "\n".join(f"- {triple}" for triple in triples))

        return {
            "matched_entities": matched,
            "subgraph_triples": triples,
            "formatted_context": "\n\n".join(context_parts),
            "max_hops": max_hops,
        }

    def get_knowledge_graph(self) -> KnowledgeGraph:
        """Get the knowledge graph."""
        return self.knowledge_graph

    def save_graph(self, path: str) -> None:
        """Export the knowledge graph to a JSON file (the index itself is stored with the chunks)."""
        self.knowledge_graph.save(path)

    def load_graph(self, path: str) -> None:
        """Load an exported graph into ``knowledge_graph`` for traversal (search keeps using the index)."""
        self.knowledge_graph.load(path)

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
