# RAG Chatbot - PDF Question Answering System

A powerful Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows users to upload PDF documents and ask questions about their content. The system uses advanced NLP techniques to provide accurate, context-aware responses based on the uploaded documents.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF files for question answering
- **Intelligent Text Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Vector Search**: Uses FAISS vector store with HuggingFace embeddings for semantic search
- **Interactive Chat Interface**: Clean, user-friendly chat interface with message history
- **Source Attribution**: View source documents and relevance scores for transparency
- **Customizable Parameters**: Adjust chunk size, overlap, and number of sources via sidebar
- **Fallback Mechanism**: Graceful fallback to general chat if document processing fails
- **Default Document Support**: Supports a default PDF (`reflexion.pdf`) for quick testing

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama3-8b-8192)
- **Embeddings**: HuggingFace BGE Embeddings (sentence-transformers/all-MiniLM-L12-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain with PyPDFLoader
- **Framework**: LangChain for RAG pipeline

## 📋 Prerequisites

- Python 3.8+
- Groq API Key
- Required Python packages (see Installation)

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install streamlit langchain-groq langchain-core langchain-community
   pip install sentence-transformers faiss-cpu pypdf torch transformers
   pip install huggingface-hub
   ```

4. **Set up your Groq API Key**:
   - Get your API key from [Groq Console](https://console.groq.com/)
   - Replace the API key in the code (line 76) or use environment variables:
   ```python
   groq_api_key = os.getenv("GROQ_API_KEY", "your-api-key-here")
   ```

## 🚀 Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

3. **Upload a PDF or use default**:
   - Upload a PDF file using the sidebar file uploader
   - Or place a file named `reflexion.pdf` in the same directory for default usage

4. **Configure settings** (optional):
   - Adjust **Chunk Size** (500-2000): Size of text chunks for processing
   - Adjust **Chunk Overlap** (50-500): Overlap between consecutive chunks
   - Set **Number of Sources** (1-10): Number of relevant sources to retrieve

5. **Start chatting**:
   - Type your questions in the chat input
   - Get AI-powered responses based on your PDF content
   - View source documents and relevance scores in expandable sections

## 📁 Project Structure

```
rag-chatbot/
│
├── app.py                 # Main Streamlit application
├── reflexion.pdf         # Default PDF (optional)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore file
└── docs/
    ├── API_REFERENCE.md # API documentation
    └── TROUBLESHOOTING.md # Common issues and solutions
```

## ⚙️ Configuration Options

### Sidebar Parameters

| Parameter | Range | Default | Description |
|-----------|--------|---------|-------------|
| Chunk Size | 500-2000 | 1000 | Size of text chunks for vector storage |
| Chunk Overlap | 50-500 | 200 | Overlap between consecutive text chunks |
| Number of Sources | 1-10 | 5 | Number of relevant sources to retrieve per query |

### Environment Variables

```bash
export GROQ_API_KEY="your-groq-api-key"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"  # If you encounter PyTorch issues
```

## 🔍 How It Works

1. **Document Processing**: PDF is loaded and split into manageable chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using HuggingFace models
3. **Vector Storage**: Embeddings are stored in FAISS vector database for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Most relevant document chunks are retrieved based on similarity
6. **Response Generation**: Groq LLM generates responses using retrieved context
7. **Source Attribution**: Original source documents and relevance scores are displayed

## 🎯 Key Features Explained

### RAG Pipeline
The application implements a complete RAG pipeline:
- **Retrieval**: Semantic search through document embeddings
- **Augmentation**: Context-aware prompt engineering
- **Generation**: LLM-powered response generation with source attribution

### Caching
Uses Streamlit's `@st.cache_resource` decorator to cache the vector store, improving performance for repeated queries.

### Error Handling
Robust error handling with fallback mechanisms:
- Graceful degradation if document processing fails
- Fallback to general chat mode when RAG pipeline encounters issues

## 🐛 Troubleshooting

### Common Issues

1. **PyTorch/Streamlit Compatibility Error**:
   ```bash
   export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
   streamlit run app.py
   ```

2. **Missing Dependencies**:
   ```bash
   pip install --upgrade streamlit torch sentence-transformers
   ```

3. **Groq API Issues**:
   - Verify your API key is correct
   - Check your Groq account credits/limits

4. **PDF Processing Errors**:
   - Ensure PDF is not password-protected
   - Check file permissions
   - Try with a different PDF file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Groq](https://groq.com/) for fast LLM inference
- [HuggingFace](https://huggingface.co/) for embeddings models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search

---

**Happy Chatting! 🤖💬**
