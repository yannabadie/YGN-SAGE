"""Neo4j driver for YGN-SAGE graph memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any
from neo4j import GraphDatabase, Driver


class Neo4jMemoryDriver:
    """Concrete implementation of GraphDatabase protocol for Neo4j."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Driver | None = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Connect to the Neo4j database."""
        if not self.driver:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.logger.info("Connected to Neo4j graph database.")

    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None

    def create_node(self, label: str, properties: dict[str, Any]) -> str:
        """Create a node with the given label and properties.
        
        Returns the ID of the created node.
        """
        self.connect()
        node_id = str(uuid.uuid4())
        properties["id"] = node_id
        
        query = f"CREATE (n:{label} $props) RETURN n.id AS id"
        
        with self.driver.session() as session:
            result = session.run(query, props=properties)
            record = result.single()
            if record:
                return record["id"]
            raise RuntimeError("Failed to create node.")

    def create_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
        """Create a relationship between two nodes by their IDs."""
        self.connect()
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        
        with self.driver.session() as session:
            session.run(query, from_id=from_id, to_id=to_id)
