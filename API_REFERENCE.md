# API Reference

## Core Functions

### `get_vectorstore(_uploaded_file=None, _chunk_size=1000, _chunk_overlap=200)`

Creates and returns a FAISS vector store from a PDF document.

**Parameters:**
- `_uploaded_file` (UploadedFile, optional): Streamlit uploaded file object
- `_chunk_size` (int): Size of text chunks (default: 1000)
- `_chunk_overlap` (int): Overlap between chunks (default: 200)

**Returns:**
- `FAISS`: Vector store object or None if creation fails

**Caching:**
- Uses `@st.cache_resource` for performance optimization
- Cache is invalidated when parameters change

### `format_docs(docs)`

Formats retrieved documents for prompt context.

**Parameters:**
- `docs` (List[Document]): List of LangChain Document objects

**Returns:**
- `str`: Formatted string with numbered sources

## Configuration Parameters

### Sidebar Controls

| Parameter | Type | Range | Default | Description |
|-----------|------|--------|---------|-------------|
| `chunk_size` | int | 500-2000 | 1000 | Text chunk size for vector storage |
| `chunk_overlap` | int | 50-500 | 200 | Overlap between consecutive chunks |
| `num_sources` | int | 1-10 | 5 | Number of sources to retrieve per query |

### Session State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `st.session_state.messages` | List[Dict] | Chat message history |

Each message dictionary contains:
- `role` (str): "user" or "assistant"
- `content` (str): Message content

## RAG Chain Components

### 1. Document Loader
```python
loader = PyPDFLoader(pdf_name)
documents = loader.load()
```

### 2. Text Splitter
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

### 3. Embeddings
```python
embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)
```

### 4. Vector Store
```python
vectorstore = FAISS.from_documents(chunks, embeddings)
```

### 5. Retriever
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": num_sources})
```

### 6. Prompt Template
```python
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the information from the context below to answer the question.
If the answer is not in the context, clearly state that the information is not available in the document.

Context from document:
{context}

Question: {question}

Instructions:
- Start with a brief summary of what you found
- Provide detailed points based on the context
- Be conversational and use phrases like 'watch ya doin' to keep it cool
- If you can't find the answer in the context, say so honestly

Answer:
""")
```

### 7. LLM Configuration
```python
groq_chat = ChatGroq(
    groq_api_key="your-api-key",
    model_name="llama3-8b-8192"
)
```

### 8. RAG Chain (LCEL)
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | groq_chat
    | StrOutputParser()
)
```

## Error Handling

### Primary Error Handling
- Vector store creation errors
- PDF loading errors
- Groq API errors
- General processing errors

### Fallback Mechanism
When RAG processing fails, the system falls back to a simple chat mode:
```python
simple_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question: {user_prompt}
Start with a summary sentence, then provide detailed points.
Talk like a cool guy using phrases like 'watch ya doin'.
""")
```

## File Operations

### PDF File Handling
- Supports uploaded files via Streamlit file uploader
- Temporary file creation for uploaded PDFs
- Default PDF support (`reflexion.pdf`)
- File existence validation

### Temporary Files
- Uploaded files are saved as `temp_uploaded.pdf`
- Automatic cleanup handled by Streamlit session management

## Vector Search Operations

### Similarity Search
```python
docs_with_scores = vectorstore.similarity_search_with_score(query, k=num_sources)
```

### Document Retrieval
```python
retrieved_docs = retriever.get_relevant_documents(query)
```

## UI Components

### Chat Interface
- Message display with role-based styling
- Real-time message appending
- Message history persistence

### Expandable Sections
- Source document viewer
- Relevance score display
- Metadata information

### Sidebar Controls
- File uploader
- Parameter sliders
- Status indicators

## Performance Optimizations

### Caching Strategy
- Vector store caching with `@st.cache_resource`
- Embedding model caching
- Parameter-based cache invalidation

### Memory Management
- Temporary file cleanup
- Efficient document chunking
- Optimized vector operations

## Security Considerations

### API Key Management
- Environment variable support
- Secure key storage recommendations
- API key rotation best practices

### File Upload Security
- PDF format validation
- File size limitations (handled by Streamlit)
- Temporary file management