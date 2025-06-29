"""
Local Embedding Module using Sentence Transformers

This module handles text embedding using Sentence Transformers.
Provides functionality to convert text chunks into vector embeddings locally.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbedder:
    """Handles text embedding using Sentence Transformers locally."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the local embedder.
        
        Args:
            model_name: Name of the Sentence Transformers model to use
        """
        self.model_name = model_name
        self.model = None
        
        try:
            logger.info(f"Loading Sentence Transformers model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
            logger.info(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformers model {model_name}: {e}")
            raise
    
    def embed_text(self, text: str, retry_count: int = 3) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure (kept for compatibility)
            
        Returns:
            List of embedding values
        """
        try:
            # Clean and prepare text
            cleaned_text = self._preprocess_text(text)
            
            # Get embedding
            if self.model is None:
                raise Exception("Model not initialized")
            
            # Convert to list of floats
            embedding = self.model.encode(cleaned_text, convert_to_numpy=False)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if self.model is None:
            raise Exception("Model not initialized")
        
        all_embeddings = []
        
        # Preprocess all texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        # Process in batches
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(cleaned_texts) + batch_size - 1)//batch_size}")
            
            try:
                # Encode the batch
                batch_embeddings = self.model.encode(batch, convert_to_numpy=False)
                
                # Convert to list of lists
                for embedding in batch_embeddings:
                    all_embeddings.append(embedding.tolist())
                    
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                # Add zero vectors as fallback for failed batch
                dimension = self.get_embedding_dimension()
                for _ in batch:
                    all_embeddings.append([0.0] * dimension)
        
        logger.info(f"Successfully embedded {len(all_embeddings)} texts")
        return all_embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (Sentence Transformers can handle long texts, but let's be conservative)
        max_length = 25000  # Increased from 10000 to accommodate very long technical explanations
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.model is None:
            return 768  # Default for most models
        dimension = self.model.get_sentence_embedding_dimension()
        return dimension if dimension is not None else 768
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding is properly formatted.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not embedding:
            return False
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False
        
        # Check for NaN or infinite values
        if any(not (x == x) or not (x != float('inf')) for x in embedding):
            return False
        
        return True
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def batch_similarity(self, query_embedding: List[float], 
                        candidate_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate similarity between a query embedding and multiple candidate embeddings.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for candidate in candidate_embeddings:
            similarity = self.cosine_similarity(query_embedding, candidate)
            similarities.append(similarity)
        return similarities
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = self.batch_similarity(query_embedding, candidate_embeddings)
        
        # Create list of (index, similarity) tuples
        indexed_similarities = list(enumerate(similarities))
        
        # Sort by similarity (descending)
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_similarities[:top_k]

# Alias for backward compatibility
GeminiEmbedder = LocalEmbedder 