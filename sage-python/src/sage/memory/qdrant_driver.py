"""Qdrant driver for YGN-SAGE vector memory."""
from __future__ import annotations

import logging
import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


class QdrantMemoryDriver:
    """Concrete implementation of VectorDatabase protocol for Qdrant."""

    def __init__(self, host: str = "localhost", port: int = 6333, vector_size: int = 1536):
        self.host = host
        self.port = port
        self.vector_size = vector_size
        self.client: QdrantClient | None = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Connect to Qdrant."""
        if not self.client:
            self.client = QdrantClient(host=self.host, port=self.port)
            self.logger.info("Connected to Qdrant vector database.")

    def close(self) -> None:
        """Close the connection."""
        if self.client:
            self.client.close()
            self.client = None

    def upsert(self, collection: str, text: str, metadata: dict[str, Any]) -> str:
        """Upsert a text (embedded via dummy/placeholder) into Qdrant.
        
        In a real implementation, `text` should be embedded using an LLM.
        """
        self.connect()
        assert self.client is not None
        
        # Ensure collection exists
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

        # Real embedding using google-genai
        try:
            from google import genai
            import os
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            genai_client = genai.Client(api_key=api_key)
            result = genai_client.models.embed_content(
                model='text-embedding-004',
                contents=text,
            )
            vector = result.embeddings[0].values
        except Exception as e:
            self.logger.error(f"Embedding failed, falling back to zeros: {e}")
            vector = [0.0] * 768

        point_id = str(uuid.uuid4())        
        # Add text to metadata for retrieval
        full_metadata = metadata.copy()
        full_metadata["text"] = text
        
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=full_metadata
                )
            ]
        )
        
        return point_id
