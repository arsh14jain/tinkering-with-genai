"""
RAG System Module

This module orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline.
Combines parsing, embedding, vector storage, and generation components.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from parser import ChatGPTParser, ConversationChunk
from embedder import GeminiEmbedder
from vector_store import ChromaVectorStore
from generator import GeminiGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system that orchestrates all components."""
    
    def __init__(self, gemini_api_key: str, 
                 chroma_persist_dir: str = "./chroma_db",
                 collection_name: str = "chat_history"):
        """
        Initialize the RAG system.
        
        Args:
            gemini_api_key: Google API key for Gemini (for text generation)
            chroma_persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
        """
        self.gemini_api_key = gemini_api_key
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        
        # Initialize components
        self.parser = ChatGPTParser()
        # Use local embedder (no API key needed)
        self.embedder = GeminiEmbedder()
        self.vector_store = ChromaVectorStore(chroma_persist_dir, collection_name, embedder=self.embedder)
        # Use API key for text generation
        self.generator = GeminiGenerator(gemini_api_key)
        
        logger.info("RAG system initialized successfully")
    
    def load_and_process_chat_data(self, chat_file_path: str, 
                                  chunk_strategy: str = "message_pairs",
                                  export_chunks: bool = False) -> List[ConversationChunk]:
        """
        Load and process ChatGPT exported data.
        
        Args:
            chat_file_path: Path to the chat.json file
            chunk_strategy: Strategy for chunking conversations
            export_chunks: Whether to export chunks to JSON for debugging
            
        Returns:
            List of processed conversation chunks
        """
        try:
            # Load and parse chat data
            logger.info(f"Loading chat data from: {chat_file_path}")
            chat_data = self.parser.load_chat_data(chat_file_path)
            
            # Parse conversations
            messages = self.parser.parse_conversations(chat_data)
            
            # Chunk conversations
            chunks = self.parser.chunk_conversations(messages, chunk_strategy)
            
            # Export chunks if requested
            if export_chunks:
                export_path = f"chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.parser.export_chunks_to_json(export_path)
            
            logger.info(f"Processed {len(chunks)} chunks from chat data")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load and process chat data: {e}")
            raise
    
    def embed_and_store_chunks(self, chunks: List[ConversationChunk], 
                              batch_size: int = 10) -> None:
        """
        Embed chunks and store them in the vector database.
        
        Args:
            chunks: List of conversation chunks to process
            batch_size: Batch size for embedding
        """
        try:
            # Prepare chunks for embedding
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "conversation_id": chunk.conversation_id,
                    "timestamp": chunk.timestamp.isoformat(),
                    "text": chunk.text,
                    "role": chunk.role,
                    "metadata": chunk.metadata
                }
                chunk_dicts.append(chunk_dict)
            
            # Extract texts for embedding
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = self.embedder.embed_batch(texts, batch_size)
            
            # Store in vector database
            logger.info("Storing chunks in vector database")
            self.vector_store.add_chunks(chunk_dicts, embeddings)
            
            logger.info(f"Successfully embedded and stored {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to embed and store chunks: {e}")
            raise
    
    def query(self, user_query: str, n_results: int = 5, 
              conversation_filter: Optional[str] = None,
              time_filter: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a user query and generate a response.
        
        Args:
            user_query: User's query
            n_results: Number of context chunks to retrieve
            conversation_filter: Optional conversation ID filter
            time_filter: Optional time range filter
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Prepare search filters
            where_filter = None
            if conversation_filter:
                where_filter = {"conversation_id": conversation_filter}
            elif time_filter:
                where_filter = {
                    "$and": [
                        {"timestamp": {"$gte": time_filter["start"]}},
                        {"timestamp": {"$lte": time_filter["end"]}}
                    ]
                }
            
            # Search for relevant context
            logger.info(f"Searching for context relevant to: '{user_query[:50]}...'")
            context_results = self.vector_store.search_by_text(
                user_query, n_results, where_filter
            )
            
            if not context_results:
                logger.warning("No relevant context found")
                return {
                    "response": "I don't have enough relevant context from your chat history to answer this question.",
                    "context_used": [],
                    "similarity_scores": [],
                    "query": user_query
                }
            
            # Generate response
            logger.info("Generating response using retrieved context")
            response = self.generator.generate_response(user_query, context_results)
            
            # Extract metadata
            context_texts = [result["text"] for result in context_results]
            similarity_scores = [result["similarity"] for result in context_results]
            
            return {
                "response": response,
                "context_used": context_texts,
                "similarity_scores": similarity_scores,
                "query": user_query,
                "num_context_chunks": len(context_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "response": "I encountered an error while processing your query. Please try again.",
                "context_used": [],
                "similarity_scores": [],
                "query": user_query,
                "error": str(e)
            }
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dictionary containing conversation summary and metadata
        """
        try:
            # Get conversation chunks
            chunks = self.vector_store.get_by_conversation_id(conversation_id)
            
            if not chunks:
                return {
                    "conversation_id": conversation_id,
                    "summary": "No conversation found with this ID.",
                    "num_chunks": 0
                }
            
            # Generate summary
            summary = self.generator.summarize_context(chunks)
            
            return {
                "conversation_id": conversation_id,
                "summary": summary,
                "num_chunks": len(chunks),
                "chunks": chunks
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {
                "conversation_id": conversation_id,
                "summary": "Error generating summary.",
                "error": str(e)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get parser stats
            parser_stats = {
                "total_chunks": len(self.parser.chunks),
                "unique_conversations": len(self.parser.get_unique_conversations()) if self.parser.chunks else 0
            }
            
            # Combine stats
            stats = {
                "vector_store": vector_stats,
                "parser": parser_stats,
                "embedding_dimension": self.embedder.get_embedding_dimension(),
                "system_components": {
                    "parser": "ChatGPTParser",
                    "embedder": "GeminiEmbedder",
                    "vector_store": "ChromaVectorStore",
                    "generator": "GeminiGenerator"
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def reset_system(self) -> None:
        """Reset the entire system (clear vector store)."""
        try:
            self.vector_store.reset_collection()
            self.parser.chunks = []
            logger.info("System reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset system: {e}")
            raise
    
    def export_system_data(self, export_dir: str = "./exports") -> None:
        """
        Export system data for backup or analysis.
        
        Args:
            export_dir: Directory to export data to
        """
        try:
            os.makedirs(export_dir, exist_ok=True)
            
            # Export chunks
            if self.parser.chunks:
                chunks_path = os.path.join(export_dir, f"chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                self.parser.export_chunks_to_json(chunks_path)
            
            # Export system stats
            stats = self.get_system_stats()
            stats_path = os.path.join(export_dir, f"system_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"System data exported to: {export_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export system data: {e}")
            raise 