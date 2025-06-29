# ChatGPT RAG System

A Retrieval-Augmented Generation (RAG) system that allows you to query your exported ChatGPT conversation history using local embeddings and Google's Gemini AI. The system uses ChromaDB as a vector database to store and retrieve relevant conversation chunks.

## Features

- Parse ChatGPT exported data
- Multiple chunking strategies (message pairs, individual messages, sliding windows)
- Local embeddings using Sentence Transformers
- ChromaDB vector storage for similarity search
- AI-powered responses using Gemini
- Advanced search with filtering by conversation ID or time range

## Requirements

- Python 3.8+
- Google Gemini API key (for text generation)
- ChatGPT exported data (conversations.json)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd tinkering-with-genai
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Quick Start

1. Export your ChatGPT data from [chat.openai.com](https://chat.openai.com) (Settings → Export Data)

2. Load and query your data:
   ```bash
   # Load data and start interactive mode
   python src/main.py --load conversations.json --interactive
   
   # Or query directly
   python src/main.py --load conversations.json --query "What did we discuss about Python?"
   ```

## Usage

### Interactive Mode
```bash
python src/main.py --load conversations.json --interactive
```

Available commands:
- `query <text>` - Ask a question about your chat history
- `stats` - Show system statistics
- `quit` - Exit the system

### Command Line Options
```bash
# Basic usage
python src/main.py --load conversations.json --query "Your question here"

# Advanced options
python src/main.py --load conversations.json --chunk-strategy message_pairs --n-results 10
```

## Project Structure

```
src/
├── parser.py          # ChatGPT data parsing and chunking
├── embedder.py        # Local embedding functionality
├── vector_store.py    # ChromaDB vector storage
├── generator.py       # Gemini text generation
├── rag_system.py      # Main RAG system orchestrator
└── main.py           # CLI interface
```

## Example Queries

- "What did we discuss about machine learning?"
- "Show me conversations about Python programming"
- "What coding problems did we solve together?"
- "Summarize our discussion about data science"
