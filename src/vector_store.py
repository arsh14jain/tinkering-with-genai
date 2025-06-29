"""
ChromaDB Vector Store Module

This module handles storing and retrieving embeddings using ChromaDB.
Provides functionality to store conversation chunks with metadata and perform similarity search.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "chat_history", embedder=None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            embedder: Embedder instance for generating embeddings (optional)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = embedder
        self.client = None
        self.collection = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Initialized ChromaDB client with persist directory: {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _initialize_collection(self):
        """Initialize or get existing collection."""
        if self.client is None:
            raise Exception("ChromaDB client not initialized")
            
        try:
            # Try to get existing collection (without embedding function for now)
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "ChatGPT conversation history embeddings"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """
        Add conversation chunks with their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors corresponding to chunks
        """
        if self.collection is None:
            raise Exception("ChromaDB collection not initialized")
            
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.get("id", chunk.get("chunk_id", f"chunk_{i}")) for i, chunk in enumerate(chunks)]
            texts = [chunk["text"] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                if "conversation_id" not in metadata:
                    metadata["conversation_id"] = chunk.get("conversation_id", "unknown")
                if "timestamp" not in metadata:
                    metadata["timestamp"] = chunk.get("timestamp", "")
                if "role" not in metadata:
                    metadata["role"] = chunk.get("role", "unknown")
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,  # type: ignore
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            raise
    
    def search(self, query_embedding: List[float], n_results: int = 5, 
               where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional filter conditions for metadata
            
        Returns:
            List of search results with documents and metadata
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        "similarity": 1.0 - (results["distances"][0][i] if results["distances"] else 0.0)
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []
    
    def search_by_text(self, query_text: str, n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query (will be embedded using our embedder).
        
        Args:
            query_text: Text query to search for
            n_results: Number of results to return
            where: Optional filter conditions for metadata
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            if self.embedder is None:
                logger.error("No embedder available for text search")
                return []
            
            # Generate embedding for the query text using our embedder
            query_embedding = self.embedder.embed_text(query_text)
            
            # Search using the embedding
            return self.search(query_embedding, n_results, where)
            
        except Exception as e:
            logger.error(f"Failed to search vector store by text: {e}")
            return []
    
    def get_by_conversation_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of chunks for the conversation
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []
            
        try:
            results = self.collection.get(
                where={"conversation_id": conversation_id},
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    result = {
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} chunks for conversation: {conversation_id}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get chunks by conversation ID: {e}")
            return []
    
    def get_by_time_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        Get chunks within a time range.
        
        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            
        Returns:
            List of chunks within the time range
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return []
            
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"timestamp": {"$gte": start_time}},
                        {"timestamp": {"$lte": end_time}}
                    ]
                },
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    result = {
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} chunks in time range: {start_time} to {end_time}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get chunks by time range: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            logger.error("ChromaDB collection not initialized")
            return {}
            
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample = self.collection.get(limit=100, include=["metadatas"])
            
            # Analyze metadata
            conversation_ids = set()
            roles = set()
            
            if sample["metadatas"]:
                for metadata in sample["metadatas"]:
                    if metadata:
                        conversation_ids.add(metadata.get("conversation_id", "unknown"))
                        roles.add(metadata.get("role", "unknown"))
            
            stats = {
                "total_chunks": count,
                "unique_conversations": len(conversation_ids),
                "roles": list(roles),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        if self.client is None:
            raise Exception("ChromaDB client not initialized")
            
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting and recreating it."""
        try:
            self.delete_collection()
            self._initialize_collection()
            logger.info("Successfully reset collection")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise 