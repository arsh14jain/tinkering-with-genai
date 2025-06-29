#!/usr/bin/env python3
"""
Script to view all data stored in ChromaDB
"""

import logging
import chromadb
from chromadb.config import Settings
from src.embedder import LocalEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def view_chroma_data():
    """View all data stored in ChromaDB."""
    
    try:
        print("🔍 Exploring ChromaDB data...")
        
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # List all collections
        collections = client.list_collections()
        print(f"\n📚 Found {len(collections)} collections:")
        for collection in collections:
            print(f"  • {collection.name}")
        
        if not collections:
            print("❌ No collections found in ChromaDB")
            return False
        
        # Get the main collection (assuming it's the first one)
        collection_name = collections[0].name
        collection = client.get_collection(name=collection_name)
        
        print(f"\n📖 Exploring collection: '{collection_name}'")
        
        # Get collection info
        count = collection.count()
        print(f"📊 Total documents: {count}")
        
        if count == 0:
            print("❌ No documents found in collection")
            return False
        
        # Get all documents
        print("\n📄 Fetching all documents...")
        results = collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        documents = results['documents'] or []
        metadatas = results['metadatas'] or []
        embeddings = results['embeddings'] or []
        
        print(f"✅ Retrieved {len(documents)} documents")
        
        # Show document statistics
        print(f"\n📈 Document Statistics:")
        print(f"  • Total documents: {len(documents)}")
        print(f"  • Embedding dimension: {len(embeddings[0]) if embeddings and len(embeddings) > 0 else 'N/A'}")
        
        # Show unique conversations
        if metadatas:
            conversation_ids = set()
            roles = set()
            for metadata in metadatas:
                if metadata and 'conversation_id' in metadata:
                    conversation_ids.add(metadata['conversation_id'])
                if metadata and 'role' in metadata:
                    roles.add(metadata['role'])
            
            print(f"  • Unique conversations: {len(conversation_ids)}")
            print(f"  • Message roles: {', '.join(roles)}")
        
        # Show sample documents
        print(f"\n📝 Sample Documents (showing first 5):")
        for i in range(min(5, len(documents))):
            print(f"\n--- Document {i+1} ---")
            
            if metadatas and i < len(metadatas) and metadatas[i]:
                metadata = metadatas[i]
                print(f"Conversation ID: {metadata.get('conversation_id', 'N/A')}")
                print(f"Role: {metadata.get('role', 'N/A')}")
                print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
                print(f"Chunk Type: {metadata.get('chunk_type', 'N/A')}")
            
            # Show document content (truncated)
            content = documents[i]
            if len(content) > 200:
                print(f"Content: {content[:200]}...")
            else:
                print(f"Content: {content}")
            
            # Show embedding info
            if embeddings and i < len(embeddings):
                embedding = embeddings[i]
                print(f"Embedding: {len(embedding)} dimensions, first 5 values: {embedding[:5]}")
        
        # Show conversation breakdown
        if metadatas:
            print(f"\n🗂️  Conversation Breakdown:")
            conv_counts = {}
            for metadata in metadatas:
                if metadata and 'conversation_id' in metadata:
                    conv_id = metadata['conversation_id']
                    conv_counts[conv_id] = conv_counts.get(conv_id, 0) + 1
            
            # Show top 10 conversations by chunk count
            sorted_convs = sorted(conv_counts.items(), key=lambda x: x[1], reverse=True)
            print("Top 10 conversations by chunk count:")
            for i, (conv_id, count) in enumerate(sorted_convs[:10], 1):
                print(f"  {i}. {conv_id}: {count} chunks")
        
        # Show role breakdown
        if metadatas:
            print(f"\n👥 Role Breakdown:")
            role_counts = {}
            for metadata in metadatas:
                if metadata and 'role' in metadata:
                    role = metadata['role']
                    role_counts[role] = role_counts.get(role, 0) + 1
            
            for role, count in role_counts.items():
                print(f"  • {role}: {count} chunks")
        
        # Test search functionality
        print(f"\n🔍 Testing Search Functionality:")
        test_queries = [
            "Python",
            "JavaScript", 
            "error",
            "debugging"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            try:
                # Use the embedder to get query embedding
                embedder = LocalEmbedder()
                query_embedding = embedder.embed_text(query)
                
                # Search in collection
                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,
                    include=['documents', 'metadatas', 'distances']
                )
                
                if search_results['documents'] and search_results['documents'][0]:
                    for j, doc in enumerate(search_results['documents'][0]):
                        distance = search_results['distances'][0][j] if search_results['distances'] else 'N/A'
                        print(f"  {j+1}. Distance: {distance:.4f}")
                        print(f"     Content: {doc[:100]}...")
                else:
                    print("  No results found")
                    
            except Exception as e:
                print(f"  Error searching for '{query}': {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to view ChromaDB data: {e}")
        return False

if __name__ == "__main__":
    success = view_chroma_data()
    if success:
        print("\n🎉 ChromaDB exploration completed!")
        print("\n💡 Tips:")
        print("  • Use this script to understand what's stored in your database")
        print("  • Check the conversation breakdown to see which chats have the most content")
        print("  • Use the search test to verify your embeddings are working")
        print("  • The embedding dimensions should match your model (768 for all-mpnet-base-v2)")
    else:
        print("\n❌ ChromaDB exploration failed!") 