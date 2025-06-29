"""
ChatGPT Data Parser

This module handles parsing of exported ChatGPT chat history from chat.json files.
Extracts conversation data and chunks it into useful units for RAG processing.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message with metadata."""
    conversation_id: str
    timestamp: datetime
    role: str  # 'user' or 'assistant'
    content: str
    message_id: str

@dataclass
class ConversationChunk:
    """Represents a chunk of conversation for RAG processing."""
    chunk_id: str
    conversation_id: str
    timestamp: datetime
    text: str
    role: str
    metadata: Dict[str, Any]

class ChatGPTParser:
    """Parser for ChatGPT exported chat history."""
    
    def __init__(self):
        self.chunks: List[ConversationChunk] = []
    
    def load_chat_data(self, file_path: str) -> Any:
        """
        Load ChatGPT exported data from JSON file.
        
        Args:
            file_path: Path to the chat.json file
            
        Returns:
            Parsed JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded chat data from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
    
    def parse_conversations(self, chat_data: Any) -> List[ChatMessage]:
        """
        Parse conversations from chat data into structured messages.
        
        Args:
            chat_data: Loaded chat data from JSON (can be list or dict)
        
        Returns:
            List of parsed chat messages
        """
        messages = []
        # If chat_data is a list, treat each item as a conversation
        if isinstance(chat_data, list):
            conversations = chat_data
        else:
            # Handle different possible structures in chat.json
            conversations = chat_data.get('conversations', [])
            if not conversations:
                conversations = chat_data.get('data', [])
        
        for conv in conversations:
            conversation_id = conv.get('id', 'unknown')
            
            # Handle ChatGPT export format with 'mapping' field
            if 'mapping' in conv:
                mapping = conv.get('mapping', {})
                for msg_id, msg_data in mapping.items():
                    try:
                        # Extract the actual message from the nested structure
                        if msg_data and isinstance(msg_data, dict) and 'message' in msg_data:
                            msg = msg_data['message']
                        else:
                            continue
                        
                        # Skip system messages or empty messages
                        if not msg or msg.get('author', {}).get('role') == 'system':
                            continue
                        
                        # Parse timestamp
                        timestamp_str = msg.get('create_time', '')
                        if timestamp_str:
                            try:
                                # Convert Unix timestamp to datetime
                                timestamp = datetime.fromtimestamp(timestamp_str)
                            except (ValueError, OSError) as e:
                                logger.warning(f"Invalid timestamp {timestamp_str} for message {msg_id}: {e}")
                                timestamp = datetime.now()
                        else:
                            timestamp = datetime.now()
                        
                        # Extract message content
                        content_obj = msg.get('content', {})
                        content = ""
                        if isinstance(content_obj, dict) and 'parts' in content_obj:
                            parts = content_obj['parts']
                            if isinstance(parts, list):
                                content = ' '.join([str(part) for part in parts if part])
                            else:
                                content = str(parts)
                        elif isinstance(content_obj, str):
                            content = content_obj
                        
                        # Get role from author
                        author = msg.get('author', {})
                        role = author.get('role', 'unknown')
                        
                        # Create message object
                        message = ChatMessage(
                            conversation_id=conversation_id,
                            timestamp=timestamp,
                            role=role,
                            content=content,
                            message_id=msg_id
                        )
                        if message.content.strip():  # Only add non-empty messages
                            messages.append(message)
                    except Exception as e:
                        logger.warning(f"Failed to parse message {msg_id} in conversation {conversation_id}: {e}")
                        continue
            
            # Handle traditional format with 'messages' field
            else:
                conv_messages = conv.get('messages', [])
                for msg in conv_messages:
                    try:
                        # Parse timestamp
                        timestamp_str = msg.get('timestamp', '')
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.now()
                        # Extract message content
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            # Handle structured content (e.g., with markdown)
                            content = ' '.join([part.get('text', '') for part in content if isinstance(part, dict)])
                        # Create message object
                        message = ChatMessage(
                            conversation_id=conversation_id,
                            timestamp=timestamp,
                            role=msg.get('role', 'unknown'),
                            content=content,
                            message_id=msg.get('id', f"{conversation_id}_{len(messages)}")
                        )
                        if message.content.strip():  # Only add non-empty messages
                            messages.append(message)
                    except Exception as e:
                        logger.warning(f"Failed to parse message in conversation {conversation_id}: {e}")
                        continue
        
        logger.info(f"Parsed {len(messages)} messages from {len(conversations)} conversations")
        return messages
    
    def chunk_conversations(self, messages: List[ChatMessage], 
                          chunk_strategy: str = "message_pairs") -> List[ConversationChunk]:
        """
        Chunk conversations into useful units for RAG processing.
        
        Args:
            messages: List of parsed chat messages
            chunk_strategy: Strategy for chunking ('message_pairs', 'individual', 'sliding_window')
            
        Returns:
            List of conversation chunks
        """
        chunks = []
        
        if chunk_strategy == "message_pairs":
            chunks = self._chunk_message_pairs(messages)
        elif chunk_strategy == "individual":
            chunks = self._chunk_individual_messages(messages)
        elif chunk_strategy == "sliding_window":
            chunks = self._chunk_sliding_window(messages)
        else:
            logger.warning(f"Unknown chunk strategy: {chunk_strategy}. Using individual messages.")
            chunks = self._chunk_individual_messages(messages)
        
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks using strategy: {chunk_strategy}")
        return chunks
    
    def _chunk_message_pairs(self, messages: List[ChatMessage]) -> List[ConversationChunk]:
        """Chunk messages into user-assistant pairs."""
        chunks = []
        
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                # Create combined text
                combined_text = f"User: {user_msg.content}\n\nAssistant: {assistant_msg.content}"
                
                chunk = ConversationChunk(
                    chunk_id=f"{user_msg.conversation_id}_pair_{i//2}",
                    conversation_id=user_msg.conversation_id,
                    timestamp=user_msg.timestamp,
                    text=combined_text,
                    role="pair",
                    metadata={
                        "user_message_id": user_msg.message_id,
                        "assistant_message_id": assistant_msg.message_id,
                        "chunk_type": "message_pair"
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_individual_messages(self, messages: List[ChatMessage]) -> List[ConversationChunk]:
        """Chunk each message individually."""
        chunks = []
        
        for msg in messages:
            chunk = ConversationChunk(
                chunk_id=msg.message_id,
                conversation_id=msg.conversation_id,
                timestamp=msg.timestamp,
                text=msg.content,
                role=msg.role,
                metadata={
                    "chunk_type": "individual_message",
                    "original_message_id": msg.message_id
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sliding_window(self, messages: List[ChatMessage], 
                            window_size: int = 3) -> List[ConversationChunk]:
        """Chunk messages using sliding window approach."""
        chunks = []
        
        for i in range(len(messages) - window_size + 1):
            window_messages = messages[i:i + window_size]
            
            # Combine messages in window
            combined_text = "\n\n".join([
                f"{msg.role.title()}: {msg.content}" 
                for msg in window_messages
            ])
            
            chunk = ConversationChunk(
                chunk_id=f"{window_messages[0].conversation_id}_window_{i}",
                conversation_id=window_messages[0].conversation_id,
                timestamp=window_messages[0].timestamp,
                text=combined_text,
                role="window",
                metadata={
                    "chunk_type": "sliding_window",
                    "window_size": window_size,
                    "start_index": i,
                    "message_ids": [msg.message_id for msg in window_messages]
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chunks_by_conversation(self, conversation_id: str) -> List[ConversationChunk]:
        """Get all chunks for a specific conversation."""
        return [chunk for chunk in self.chunks if chunk.conversation_id == conversation_id]
    
    def get_chunks_by_time_range(self, start_time: datetime, 
                                end_time: datetime) -> List[ConversationChunk]:
        """Get chunks within a specific time range."""
        return [
            chunk for chunk in self.chunks 
            if start_time <= chunk.timestamp <= end_time
        ]
    
    def get_unique_conversations(self) -> List[str]:
        """Get list of unique conversation IDs."""
        return list(set(chunk.conversation_id for chunk in self.chunks))
    
    def export_chunks_to_json(self, file_path: str) -> None:
        """Export chunks to JSON file for debugging or backup."""
        chunk_data = []
        for chunk in self.chunks:
            chunk_data.append({
                "chunk_id": chunk.chunk_id,
                "conversation_id": chunk.conversation_id,
                "timestamp": chunk.timestamp.isoformat(),
                "text": chunk.text,
                "role": chunk.role,
                "metadata": chunk.metadata
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(chunk_data)} chunks to {file_path}") 