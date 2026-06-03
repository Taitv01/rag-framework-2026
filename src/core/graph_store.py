"""
Graph Store
===========

Persistent graph storage backends for KnowledgeGraph.

Supported backends:
- Neo4j (production, persistent, Cypher queries)
- NetworkX (in-memory, already in graph_rag.py)

Usage:
    from src.core.graph_store import Neo4jBackend

    # Connect to Neo4j
    backend = Neo4jBackend(uri="bolt://localhost:7687", user="neo4j", password="password")
    backend.connect()

    # Sync entities and relationships
    backend.sync_entity(entity)
    backend.sync_relationship(relationship)

    # Query with Cypher
    results = backend.query_cypher("MATCH (n:Entity) RETURN n LIMIT 10")

    # Close connection
    backend.close()
"""

import logging
import json
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class Neo4jBackend:
    """
    Neo4j backend for KnowledgeGraph persistence.

    Provides persistent graph storage that supplements the in-memory
    NetworkX graph. Supports Cypher queries for complex traversals.

    Example:
        backend = Neo4jBackend(uri="bolt://localhost:7687", user="neo4j", password="pass")
        backend.connect()

        # Sync an entity
        backend.sync_entity(Entity(name="Thạch Sanh", entity_type="PERSON", description="..."))

        # Cypher query
        results = backend.query_cypher(
            "MATCH (p:Entity {name: $name})-[:RELATES_TO]->(other) RETURN other",
            params={"name": "Thạch Sanh"}
        )

        backend.close()
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
        database: str = "neo4j",
    ):
        """
        Initialize Neo4j backend.

        Args:
            uri: Neo4j bolt URI
            user: Neo4j username
            password: Neo4j password (or from NEO4J_PASSWORD env var)
            database: Neo4j database name
        """
        self.uri = uri
        self.user = user
        self._password = password
        self.database = database
        self._driver = None
        self._connected = False

    def _get_password(self) -> str:
        """Get password from param or environment."""
        if self._password:
            return self._password
        import os
        return os.getenv("NEO4J_PASSWORD", "")

    def connect(self) -> bool:
        """
        Connect to Neo4j.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self._get_password()),
            )
            # Verify connection
            self._driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except ImportError:
            logger.warning(
                "neo4j package not installed. Install with: pip install neo4j"
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self._connected = False
            return False

    def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False
            logger.info("Neo4j connection closed")

    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        if not self._connected or not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            self._connected = False
            return False

    def sync_entity(self, entity) -> None:
        """
        Sync an entity to Neo4j (upsert).

        Args:
            entity: Entity object with name, entity_type, description
        """
        if not self.is_connected():
            return

        query = """
        MERGE (e:Entity {name: $name})
        SET e.entity_type = $entity_type,
            e.description = $description
        """
        self._execute_write(query, {
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
        })

    def sync_relationship(self, relationship) -> None:
        """
        Sync a relationship to Neo4j (upsert).

        Args:
            relationship: Relationship object with source, target, relationship_type, description
        """
        if not self.is_connected():
            return

        # Ensure both entities exist
        for name in [relationship.source, relationship.target]:
            self._execute_write(
                "MERGE (e:Entity {name: $name})",
                {"name": name},
            )

        # Create relationship
        rel_type = relationship.relationship_type.upper().replace(" ", "_")
        query = f"""
        MATCH (a:Entity {{name: $source}})
        MATCH (b:Entity {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.description = $description,
            r.weight = $weight
        """
        self._execute_write(query, {
            "source": relationship.source,
            "target": relationship.target,
            "description": relationship.description,
            "weight": relationship.weight,
        })

    def query_cypher(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            List of result records as dicts
        """
        if not self.is_connected():
            return []

        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []

    def get_entity_neighbors(
        self, name: str, depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring entities via Cypher.

        Args:
            name: Entity name
            depth: Number of hops

        Returns:
            List of neighbor entity dicts
        """
        query = """
        MATCH (e:Entity {name: $name})-[r*1..""" + str(depth) + """]->(neighbor:Entity)
        RETURN DISTINCT neighbor.name AS name,
               neighbor.entity_type AS entity_type,
               neighbor.description AS description
        """
        return self.query_cypher(query, {"name": name})

    def find_paths(
        self, start: str, end: str, max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two entities.

        Args:
            start: Start entity name
            end: End entity name
            max_depth: Maximum path length

        Returns:
            List of paths (each path is a list of entity names)
        """
        query = """
        MATCH path = (start:Entity {name: $start})-[*1..""" + str(max_depth) + """]->(end:Entity {name: $end})
        RETURN [n IN nodes(path) | n.name] AS path
        LIMIT 10
        """
        results = self.query_cypher(query, {"start": start, "end": end})
        return [r["path"] for r in results]

    def get_entity_count(self) -> int:
        """Get total number of entities."""
        results = self.query_cypher("MATCH (e:Entity) RETURN count(e) AS count")
        return results[0]["count"] if results else 0

    def get_relationship_count(self) -> int:
        """Get total number of relationships."""
        results = self.query_cypher("MATCH ()-[r]->() RETURN count(r) AS count")
        return results[0]["count"] if results else 0

    def clear(self) -> None:
        """Delete all entities and relationships."""
        if self.is_connected():
            self._execute_write("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j graph cleared")

    def sync_from_knowledge_graph(self, kg) -> int:
        """
        Sync an entire KnowledgeGraph to Neo4j.

        Args:
            kg: KnowledgeGraph instance

        Returns:
            Number of entities synced
        """
        if not self.is_connected():
            return 0

        count = 0
        for entity in kg.entities.values():
            self.sync_entity(entity)
            count += 1

        for rel in kg.relationships:
            self.sync_relationship(rel)

        logger.info(f"Synced {count} entities and {len(kg.relationships)} relationships to Neo4j")
        return count

    def _execute_write(self, query: str, params: Optional[Dict] = None):
        """Execute a write query."""
        try:
            with self._driver.session(database=self.database) as session:
                session.execute_write(
                    lambda tx: tx.run(query, params or {})
                )
        except Exception as e:
            logger.error(f"Write query failed: {e}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
