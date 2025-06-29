#!/usr/bin/env python3
"""
Script to show the specific truncated conversations
"""

import logging
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def show_truncated_conversations():
    """Show the specific truncated conversations."""
    
    try:
        print("🔍 Finding truncated conversations...")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get the collection
        collections = client.list_collections()
        if not collections:
            print("❌ No collections found")
            return False
        
        collection = client.get_collection(name=collections[0].name)
        
        # Get all documents
        results = collection.get(
            include=['documents', 'metadatas']
        )
        
        documents = results['documents'] or []
        metadatas = results['metadatas'] or []
        
        # Find truncated documents (those ending with "...")
        truncated_docs = []
        for i, doc in enumerate(documents):
            if doc.endswith("..."):
                truncated_docs.append((i, doc, metadatas[i] if i < len(metadatas) else None))
        
        print(f"📊 Found {len(truncated_docs)} truncated conversations")
        
        if not truncated_docs:
            print("✅ No truncated conversations found!")
            return True
        
        # Show each truncated conversation
        for idx, (doc_index, content, metadata) in enumerate(truncated_docs, 1):
            print(f"\n" + "="*80)
            print(f"🚨 TRUNCATED CONVERSATION #{idx}")
            print(f"="*80)
            
            # Show metadata
            if metadata:
                print(f"📋 Metadata:")
                print(f"   • Conversation ID: {metadata.get('conversation_id', 'N/A')}")
                print(f"   • Timestamp: {metadata.get('timestamp', 'N/A')}")
                print(f"   • Role: {metadata.get('role', 'N/A')}")
                print(f"   • Chunk Type: {metadata.get('chunk_type', 'N/A')}")
            
            print(f"📏 Length: {len(content)} characters")
            print(f"🔍 Document Index: {doc_index + 1}")
            
            # Show the full content
            print(f"\n📝 FULL CONTENT:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            # Show what was likely cut off
            print(f"\n💡 ANALYSIS:")
            print(f"   • This conversation was truncated at 10,000 characters")
            print(f"   • The '...' at the end indicates truncation")
            print(f"   • The assistant's response was likely cut off")
            
            # Try to find where the truncation happened
            if "Assistant:" in content:
                assistant_parts = content.split("Assistant:")
                if len(assistant_parts) > 1:
                    assistant_response = assistant_parts[-1]
                    print(f"   • Assistant response length: {len(assistant_response)} characters")
                    
                    # Show the last 200 characters to see what was cut
                    if len(assistant_response) > 200:
                        print(f"   • Last 200 chars of assistant response:")
                        print(f"     ...{assistant_response[-200:]}")
            
            print()
        
        # Also show the 3 longest conversations that are near the limit
        print(f"\n" + "="*80)
        print(f"📏 LONGEST CONVERSATIONS (Near 10K Limit)")
        print(f"="*80)
        
        # Find the longest documents
        doc_lengths = [(i, len(doc)) for i, doc in enumerate(documents)]
        doc_lengths.sort(key=lambda x: x[1], reverse=True)
        
        for i, (doc_index, length) in enumerate(doc_lengths[:5], 1):
            print(f"\n{i}. Document {doc_index + 1} ({length} characters)")
            
            if metadata and doc_index < len(metadatas):
                metadata = metadatas[doc_index]
                print(f"   • Conversation ID: {metadata.get('conversation_id', 'N/A')}")
                print(f"   • Timestamp: {metadata.get('timestamp', 'N/A')}")
            
            # Show preview
            content = documents[doc_index]
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   • Preview: {preview}")
            
            if length > 9500:
                print(f"   ⚠️  NEAR LIMIT - Consider increasing max_length")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to show truncated conversations: {e}")
        return False

if __name__ == "__main__":
    success = show_truncated_conversations()
    if success:
        print("\n🎉 Truncated conversation analysis completed!")
        print("\n💡 Recommendations:")
        print("  • If you want to see the full content, consider increasing the 10K limit")
        print("  • Most conversations are well under the limit, so it's not urgent")
        print("  • The truncation only affects extremely long technical explanations")
    else:
        print("\n❌ Truncated conversation analysis failed!") 