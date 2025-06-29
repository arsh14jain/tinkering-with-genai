#!/usr/bin/env python3
"""
Script to re-process data with the new 25K character limit
"""

import logging
import shutil
import os
from src.parser import ChatGPTParser
from src.embedder import LocalEmbedder
from src.vector_store import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reprocess_with_new_limit():
    """Re-process data with the new 25K character limit."""
    
    try:
        print("🔄 Re-processing data with new 25K character limit...")
        
        # Backup existing data
        if os.path.exists("./chroma_db"):
            print("📦 Backing up existing ChromaDB data...")
            shutil.copytree("./chroma_db", "./chroma_db_backup", dirs_exist_ok=True)
            print("✅ Backup created at ./chroma_db_backup")
        
        # Clear existing data
        if os.path.exists("./chroma_db"):
            print("🗑️  Clearing existing ChromaDB data...")
            shutil.rmtree("./chroma_db")
            print("✅ Cleared existing data")
        
        # Initialize components with new limit
        print("🚀 Initializing components...")
        parser = ChatGPTParser()
        embedder = LocalEmbedder()  # Now uses 25K limit
        vector_store = ChromaVectorStore("./chroma_db", "chat_history", embedder=embedder)
        
        # Load and parse data
        print("📂 Loading conversation data...")
        chat_data = parser.load_chat_data("conversations.json")
        messages = parser.parse_conversations(chat_data)
        chunks = parser.chunk_conversations(messages, chunk_strategy="message_pairs")
        
        print(f"✅ Parsed {len(messages)} messages into {len(chunks)} chunks")
        
        # Process chunks in batches
        batch_size = 50
        total_chunks = len(chunks)
        
        print(f"🔍 Processing {total_chunks} chunks with new 25K limit...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
            
            # Convert chunks to format expected by vector store
            chunk_dicts = []
            for chunk in batch:
                chunk_dict = {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "conversation_id": chunk.conversation_id,
                        "timestamp": chunk.timestamp.isoformat(),
                        "role": chunk.role,
                        **chunk.metadata
                    }
                }
                chunk_dicts.append(chunk_dict)
            
            # Create embeddings
            texts = [chunk.text for chunk in batch]
            embeddings = embedder.embed_batch(texts, batch_size=batch_size)
            
            # Store in vector store
            vector_store.add_chunks(chunk_dicts, embeddings)
        
        print("✅ Data re-processed with new 25K limit!")
        
        # Check for truncation
        print("\n🔍 Checking for truncation with new limit...")
        truncated_count = 0
        for chunk in chunks:
            if chunk.text.endswith("..."):
                truncated_count += 1
        
        print(f"📊 Results:")
        print(f"  • Total chunks: {len(chunks)}")
        print(f"  • Truncated chunks: {truncated_count}")
        print(f"  • Truncation rate: {truncated_count/len(chunks)*100:.1f}%")
        
        if truncated_count == 0:
            print("🎉 No truncation with new 25K limit!")
        else:
            print(f"⚠️  {truncated_count} chunks still truncated (these are extremely long)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to re-process data: {e}")
        return False

if __name__ == "__main__":
    success = reprocess_with_new_limit()
    if success:
        print("\n🎉 Re-processing completed!")
        print("\n💡 Next steps:")
        print("  • Run 'python check_conversation_lengths.py' to see the difference")
        print("  • Run 'python show_truncated_conversations.py' to check remaining truncation")
        print("  • Your backup is at ./chroma_db_backup if you need it")
    else:
        print("\n❌ Re-processing failed!") 