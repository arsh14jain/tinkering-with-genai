# 🤖 ChatGPT RAG System

A Retrieval-Augmented Generation (RAG) system that allows you to query your exported ChatGPT conversation history using Google's Gemini AI. The system uses ChromaDB as a vector database to store and retrieve relevant conversation chunks.

## 🚀 Features

- **📂 Parse ChatGPT Data**: Load and parse exported chat.json files from ChatGPT
- **🔧 Multiple Chunking Strategies**: Support for message pairs, individual messages, and sliding windows
- **🧠 Gemini Embeddings**: Use Google's Gemini API for text embeddings
- **🗄️ ChromaDB Storage**: Local vector database for efficient similarity search
- **🤖 AI-Powered Responses**: Generate contextual responses using Gemini
- **🔍 Advanced Search**: Filter by conversation ID or time range
- **📊 System Statistics**: Monitor your RAG system performance
- **💾 Data Export**: Backup and analyze your processed data

## 📋 Requirements

- Python 3.8+
- Google Gemini API key
- ChatGPT exported data (chat.json)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd tinkering-with-genai
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Gemini API key**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

## 📁 Project Structure

```
src/
├── parser.py          # ChatGPT data parsing and chunking
├── embedder.py        # Gemini embedding functionality
├── vector_store.py    # ChromaDB vector storage
├── generator.py       # Gemini text generation
├── rag_system.py      # Main RAG system orchestrator
├── main.py           # CLI interface
└── weather.py        # MCP weather server (separate project)

requirements.txt      # Python dependencies
README.md            # This file
```

## 🎯 Usage

### Quick Start

1. **Export your ChatGPT data**:
   - Go to [chat.openai.com](https://chat.openai.com)
   - Navigate to Settings → Export Data
   - Download your chat.json file

2. **Load and query your data**:
   ```bash
   # Load data and start interactive mode
   python src/main.py --load chat.json --interactive
   
   # Or query directly
   python src/main.py --load chat.json --query "What did we discuss about Python?"
   ```

### Interactive Mode

Start the interactive CLI:
```bash
python src/main.py --load chat.json --interactive
```

Available commands:
- `query <text>` - Ask a question about your chat history
- `stats` - Show system statistics
- `summary <conv_id>` - Get summary of a specific conversation
- `reset` - Reset the system (clear database)
- `export` - Export system data
- `quit` - Exit the system

### Command Line Options

```bash
# Basic usage
python src/main.py --load chat.json --query "Your question here"

# Advanced options
python src/main.py \
  --load chat.json \
  --chunk-strategy message_pairs \
  --n-results 10 \
  --batch-size 20 \
  --interactive

# System management
python src/main.py --stats                    # Show statistics
python src/main.py --reset                    # Reset system
python src/main.py --export ./backup          # Export data
python src/main.py --summary "conv_123"       # Get conversation summary
```

### Chunking Strategies

- **`message_pairs`** (default): Groups user-assistant message pairs
- **`individual`**: Each message as a separate chunk
- **`sliding_window`**: Overlapping windows of messages

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Command Line Arguments

- `--api-key`: Gemini API key (alternative to environment variable)
- `--chroma-dir`: ChromaDB persistence directory (default: `./chroma_db`)
- `--collection`: ChromaDB collection name (default: `chat_history`)
- `--chunk-strategy`: Chunking strategy (default: `message_pairs`)
- `--n-results`: Number of context chunks to retrieve (default: 5)
- `--batch-size`: Batch size for embedding (default: 10)

## 📊 System Components

### 1. Parser (`parser.py`)
- Loads ChatGPT exported JSON data
- Parses conversations and messages
- Implements multiple chunking strategies
- Handles different data formats

### 2. Embedder (`embedder.py`)
- Uses Gemini API for text embeddings
- Supports batch processing
- Includes retry logic and error handling
- Provides similarity calculations

### 3. Vector Store (`vector_store.py`)
- ChromaDB integration for vector storage
- Metadata filtering and search
- Conversation and time-based queries
- Collection management

### 4. Generator (`generator.py`)
- Gemini text generation with context
- Customizable prompts and parameters
- Response validation and summarization
- Keyword extraction

### 5. RAG System (`rag_system.py`)
- Orchestrates all components
- Provides high-level API
- Handles data flow and error management
- System statistics and maintenance

## 🔍 Example Queries

Once your data is loaded, you can ask questions like:

- "What did we discuss about machine learning?"
- "Show me conversations about Python programming"
- "What was the main topic of our last conversation?"
- "Summarize our discussion about data science"
- "What coding problems did we solve together?"

## 📈 Performance Tips

1. **Chunking Strategy**: Use `message_pairs` for better context preservation
2. **Batch Size**: Adjust based on your API rate limits (10-20 is usually good)
3. **Number of Results**: 5-10 context chunks usually provide good balance
4. **Storage**: ChromaDB data is persisted locally for fast subsequent queries

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `GEMINI_API_KEY` is set correctly
2. **Import Errors**: Ensure all dependencies are installed
3. **Memory Issues**: Reduce batch size for large datasets
4. **Rate Limiting**: Increase delays between API calls

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for AI capabilities
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [ChatGPT](https://chat.openai.com/) for conversation data format

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Check the documentation

---

**Happy querying! 🚀**
