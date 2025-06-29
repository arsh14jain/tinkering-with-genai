#!/usr/bin/env python3
"""
RAG System CLI

Command-line interface for the RAG (Retrieval-Augmented Generation) system.
Handles loading chat data, building the knowledge base, and interactive querying.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import json

from rag_system import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment_variables() -> Optional[str]:
    """Load Gemini API key from environment variable."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    return api_key

def interactive_mode(rag_system: RAGSystem):
    """Run the system in interactive mode."""
    print("\n" + "="*60)
    print("🤖 ChatGPT RAG System - Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  query <text>     - Ask a question")
    print("  stats            - Show system statistics")
    print("  summary <conv_id> - Get conversation summary")
    print("  reset            - Reset the system")
    print("  export           - Export system data")
    print("  quit             - Exit the system")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            elif user_input.lower() == 'stats':
                stats = rag_system.get_system_stats()
                print("\n📊 System Statistics:")
                print(json.dumps(stats, indent=2))
            
            elif user_input.lower() == 'reset':
                confirm = input("Are you sure you want to reset the system? (y/N): ")
                if confirm.lower() == 'y':
                    rag_system.reset_system()
                    print("✅ System reset successfully")
                else:
                    print("Reset cancelled")
            
            elif user_input.lower() == 'export':
                export_dir = input("Export directory (default: ./exports): ").strip() or "./exports"
                rag_system.export_system_data(export_dir)
                print(f"✅ System data exported to {export_dir}")
            
            elif user_input.lower().startswith('summary '):
                parts = user_input.split(' ', 1)
                if len(parts) == 2:
                    conv_id = parts[1]
                    summary = rag_system.get_conversation_summary(conv_id)
                    print(f"\n📝 Conversation Summary ({conv_id}):")
                    print(f"Summary: {summary['summary']}")
                    print(f"Chunks: {summary.get('num_chunks', 0)}")
                else:
                    print("❌ Please provide a conversation ID")
            
            elif user_input.lower().startswith('query '):
                parts = user_input.split(' ', 1)
                if len(parts) == 2:
                    query = parts[1]
                    print(f"\n🔍 Processing query: {query}")
                    
                    result = rag_system.query(query)
                    
                    print(f"\n🤖 Response:")
                    print(result['response'])
                    
                    if result.get('context_used'):
                        print(f"\n📚 Context used ({result['num_context_chunks']} chunks):")
                        for i, context in enumerate(result['context_used'][:3], 1):
                            print(f"  {i}. {context[:100]}...")
                    
                    if result.get('similarity_scores'):
                        avg_similarity = sum(result['similarity_scores']) / len(result['similarity_scores'])
                        print(f"\n📈 Average similarity: {avg_similarity:.3f}")
                else:
                    print("❌ Please provide a query")
            
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"❌ Error: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ChatGPT RAG System - Query your chat history using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load chat data and start interactive mode
  python main.py --load chat.json --interactive
  
  # Query the system directly
  python main.py --load chat.json --query "What did we discuss about Python?"
  
  # Get system statistics
  python main.py --stats
  
  # Reset the system
  python main.py --reset
        """
    )
    
    # Main arguments
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--load', help='Load chat data from JSON file')
    parser.add_argument('--query', help='Query the system')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--reset', action='store_true', help='Reset the system')
    parser.add_argument('--export', help='Export system data to directory')
    parser.add_argument('--summary', help='Get summary of conversation by ID')
    
    # Optional arguments
    parser.add_argument('--chunk-strategy', 
                       choices=['message_pairs', 'individual', 'sliding_window'],
                       default='message_pairs',
                       help='Strategy for chunking conversations (default: message_pairs)')
    parser.add_argument('--n-results', type=int, default=5,
                       help='Number of context chunks to retrieve (default: 5)')
    parser.add_argument('--chroma-dir', default='./chroma_db',
                       help='ChromaDB persistence directory (default: ./chroma_db)')
    parser.add_argument('--collection', default='chat_history',
                       help='ChromaDB collection name (default: chat_history)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for embedding (default: 10)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or load_environment_variables()
    if not api_key:
        print("❌ Gemini API key is required. Set GEMINI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag_system = RAGSystem(
            gemini_api_key=api_key,
            chroma_persist_dir=args.chroma_dir,
            collection_name=args.collection
        )
        
        # Load chat data if specified
        if args.load:
            if not os.path.exists(args.load):
                print(f"❌ Chat file not found: {args.load}")
                sys.exit(1)
            
            print(f"📂 Loading chat data from: {args.load}")
            chunks = rag_system.load_and_process_chat_data(
                args.load, 
                chunk_strategy=args.chunk_strategy,
                export_chunks=True
            )
            
            print(f"🔧 Embedding and storing {len(chunks)} chunks...")
            if not chunks:
                logger.warning("No conversation chunks found. Nothing to embed or store.")
                print("❌ No valid conversation chunks found in the data.")
                return
            rag_system.embed_and_store_chunks(chunks, args.batch_size)
            print("✅ Chat data loaded and processed successfully")
        
        # Handle different modes
        if args.interactive:
            interactive_mode(rag_system)
        
        elif args.query:
            print(f"🔍 Processing query: {args.query}")
            result = rag_system.query(args.query, args.n_results)
            
            print(f"\n🤖 Response:")
            print(result['response'])
            
            if result.get('context_used'):
                print(f"\n📚 Context used ({result['num_context_chunks']} chunks):")
                for i, context in enumerate(result['context_used'][:3], 1):
                    print(f"  {i}. {context[:100]}...")
        
        elif args.stats:
            stats = rag_system.get_system_stats()
            print("\n📊 System Statistics:")
            print(json.dumps(stats, indent=2))
        
        elif args.reset:
            confirm = input("Are you sure you want to reset the system? (y/N): ")
            if confirm.lower() == 'y':
                rag_system.reset_system()
                print("✅ System reset successfully")
            else:
                print("Reset cancelled")
        
        elif args.export:
            rag_system.export_system_data(args.export)
            print(f"✅ System data exported to {args.export}")
        
        elif args.summary:
            summary = rag_system.get_conversation_summary(args.summary)
            print(f"\n📝 Conversation Summary ({args.summary}):")
            print(f"Summary: {summary['summary']}")
            print(f"Chunks: {summary.get('num_chunks', 0)}")
        
        else:
            # No specific action, show help
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 