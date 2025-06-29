"""
Gemini Text Generation Module

This module handles text generation using Google's Gemini API.
Provides functionality to generate responses based on retrieved context and user queries.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiGenerator:
    """Handles text generation using Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini generator.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Name of the generation model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        try:
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Successfully initialized Gemini generator with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model {model_name}: {e}")
            raise
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], 
                         max_tokens: int = 1000, temperature: float = 0.7,
                         retry_count: int = 3) -> str:
        """
        Generate a response based on query and retrieved context.
        
        Args:
            query: User's query
            context: List of retrieved context chunks
            max_tokens: Maximum tokens for response
            temperature: Creativity level (0.0 to 1.0)
            retry_count: Number of retries on failure
            
        Returns:
            Generated response text
        """
        if self.model is None:
            logger.error("Model not initialized")
            return "I apologize, but the model is not properly initialized."
        
        for attempt in range(retry_count):
            try:
                # Construct prompt with context
                prompt = self._construct_prompt(query, context)
                
                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                
                if response and hasattr(response, 'text'):
                    return response.text
                else:
                    raise Exception("Invalid response from Gemini")
                    
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate response after {retry_count} attempts")
                    return "I apologize, but I encountered an error while generating a response. Please try again."
        
        return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _construct_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt with retrieved context and user query.
        
        Args:
            query: User's query
            context: List of retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Format context
        context_text = ""
        if context:
            context_text = "Based on the following conversation history:\n\n"
            for i, chunk in enumerate(context, 1):
                context_text += f"Context {i}:\n{chunk['text']}\n\n"
        
        # Construct the full prompt
        prompt = f"""{context_text}User Query: {query}

Please provide a helpful and relevant response based on the conversation history above. 
If the context doesn't contain relevant information for the query, please say so clearly.
Keep your response concise and focused on the user's question.

Response:"""
        
        return prompt
    
    def generate_with_system_prompt(self, query: str, context: List[Dict[str, Any]], 
                                   system_prompt: str, max_tokens: int = 1000, 
                                   temperature: float = 0.7) -> str:
        """
        Generate response with a custom system prompt.
        
        Args:
            query: User's query
            context: List of retrieved context chunks
            system_prompt: Custom system prompt
            max_tokens: Maximum tokens for response
            temperature: Creativity level
            
        Returns:
            Generated response text
        """
        if self.model is None:
            logger.error("Model not initialized")
            return "I apologize, but the model is not properly initialized."
        
        try:
            # Construct prompt with system prompt
            context_text = ""
            if context:
                context_text = "Conversation History:\n\n"
                for i, chunk in enumerate(context, 1):
                    context_text += f"Context {i}:\n{chunk['text']}\n\n"
            
            prompt = f"""{system_prompt}

{context_text}User Query: {query}

Response:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                return "I apologize, but I encountered an error while generating a response."
                
        except Exception as e:
            logger.error(f"Failed to generate response with system prompt: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def summarize_context(self, context: List[Dict[str, Any]], 
                         max_tokens: int = 500) -> str:
        """
        Generate a summary of the retrieved context.
        
        Args:
            context: List of retrieved context chunks
            max_tokens: Maximum tokens for summary
            
        Returns:
            Summary text
        """
        if self.model is None:
            logger.error("Model not initialized")
            return "I apologize, but the model is not properly initialized."
        
        if not context:
            return "No relevant context found."
        
        try:
            # Combine context text
            context_text = "\n\n".join([chunk['text'] for chunk in context])
            
            prompt = f"""Please provide a concise summary of the following conversation context:

{context_text}

Summary:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3  # Lower temperature for more factual summaries
                )
            )
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                return "Unable to generate summary."
                
        except Exception as e:
            logger.error(f"Failed to summarize context: {e}")
            return "Unable to generate summary."
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text using Gemini.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        if self.model is None:
            logger.error("Model not initialized")
            return []
        
        try:
            prompt = f"""Extract the {max_keywords} most important keywords or key phrases from the following text. 
Return only the keywords, separated by commas:

{text}

Keywords:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.1  # Low temperature for consistent extraction
                )
            )
            
            if response and hasattr(response, 'text'):
                keywords = [kw.strip() for kw in response.text.split(',')]
                return keywords[:max_keywords]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return []
    
    def validate_response(self, response: str) -> bool:
        """
        Validate that a generated response is appropriate.
        
        Args:
            response: Generated response text
            
        Returns:
            True if response is valid, False otherwise
        """
        if not response or not response.strip():
            return False
        
        # Check for error indicators
        error_indicators = [
            "i apologize", "i encountered an error", "unable to", 
            "failed to", "error occurred", "sorry, but"
        ]
        
        response_lower = response.lower()
        for indicator in error_indicators:
            if indicator in response_lower:
                return False
        
        return True 