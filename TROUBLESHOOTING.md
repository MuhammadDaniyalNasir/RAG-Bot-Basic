# Troubleshooting Guide

## Common Issues and Solutions

### 1. RuntimeError: no running event loop / PyTorch Issues

**Error:**
```
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
```

**Solutions:**

**Option A: Disable File Watcher (Recommended)**
```bash
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
streamlit run app.py
```

**Option B: Configuration File**
Create `.streamlit/config.toml`:
```toml
[server]
fileWatcherType = "none"
```

**Option C: Code-level Fix**
Add at the beginning of your script:
```python
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
```

### 2. Groq API Issues

**Error:**
```
AuthenticationError: Invalid API key
```

**Solutions:**
1. Verify your API key from [Groq Console](https://console.groq.com/)
2. Set environment variable:
   ```bash
   export GROQ_API_KEY="your-actual-api-key"
   ```
3. Update code to use environment variable:
   ```python
   groq_api_key = os.getenv("GROQ_API_KEY")
   ```

**Rate Limiting Issues:**
- Check your Groq account limits
- Implement retry logic if needed
- Consider upgrading your plan

### 3. PDF Processing Errors

**Error:**
```
No content found in the PDF
```

**Possible Causes & Solutions:**
1. **Password-protected PDF**: Remove password protection
2. **Corrupted PDF**: Try with a different file
3. **Image-only PDF**: Use OCR preprocessing
4. **Large PDF**: Reduce file size or increase memory

**Error:**
```
PDF file not found
```

**Solutions:**
1. Check file path and permissions
2. Ensure `reflexion.pdf` exists in the project directory
3. Verify uploaded file is properly saved

### 4. Memory Issues

**Error:**
```
OutOfMemoryError
```

**Solutions:**
1. Reduce chunk size in sidebar
2. Reduce number of sources to retrieve
3. Process smaller PDF files
4. Restart the application

**For Large PDFs:**
```python
# Reduce chunk size
chunk_size = 500  # instead of 1000
chunk_overlap = 100  # instead of 200
```

### 5. Embedding Model Issues

**Error:**
```
OSError: Can't load tokenizer for 'sentence-transformers/all-MiniLM-L12-v2'
```

**Solutions:**
1. Check internet connection
2. Clear HuggingFace cache:
   ```bash
   rm -rf ~/.cache/huggingface/
   ```
3. Manually download model:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
   ```

### 6. FAISS Installation Issues

**Error:**
```
ImportError: No module named 'faiss'
```

**Solutions:**

**For CPU-only:**
```bash
pip install faiss-cpu
```

**For GPU (if available):**
```bash
pip install faiss-gpu
```

**For M1/M2 Macs:**
```bash
conda install -c conda-forge faiss-cpu
```

### 7. Streamlit Issues

**Port Already in Use:**
```bash
streamlit run app.py --server.port 8502
```

**Streamlit Not Found:**
```bash
pip install --upgrade streamlit
```

**Browser Not Opening:**
```bash
streamlit run app.py --server.headless true
```
Then manually navigate to `http://localhost:8501`

### 8. Dependency Conflicts

**Error:**
```
VersionConflict: package versions incompatible
```

**Solutions:**

**Create Fresh Environment:**
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install -r requirements.txt
```

**Update All Packages:**
```bash
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### 9. Chat Interface Issues

**Messages Not Displaying:**
- Check browser console for JavaScript errors
- Clear browser cache
- Try incognito/private mode

**Session State Issues:**
- Restart the Streamlit app
- Clear session state manually:
  ```python
  if st.button("Clear Chat History"):
      st.session_state.messages = []
      st.rerun()
  ```

### 10. Performance Issues

**Slow Response Times:**
1. **Reduce chunk overlap:**
   ```python
   chunk_overlap = 50  # instead of 200
   ```

2. **Optimize number of sources:**
   ```python
   num_sources = 3  # instead of 5
   ```

3. **Use smaller embedding model:**
   ```python
   embeddings = HuggingFaceBgeEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"  # smaller model
   )
   ```

### 11. Vector Store Issues

**Error:**
```
Failed to create vector store
```

**Debugging Steps:**
1. Check if documents are loaded:
   ```python
   print(f"Loaded {len(documents)} documents")
   ```

2. Verify chunks are created:
   ```python
   print(f"Created {len(chunks)} chunks")
   ```

3. Test embeddings separately:
   ```python
   embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
   test_embedding = embeddings.embed_query("test")
   print(f"Embedding dimension: {len(test_embedding)}")
   ```

## Environment-Specific Issues

### Windows

**Path Issues:**
```python
# Use forward slashes or raw strings
pdf_path = r"C:\path\to\your\file.pdf"
# or
pdf_path = "C:/path/to/your/file.pdf"
```

**PowerShell Execution Policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS

**M1/M2 Compatibility:**
```bash
# Use conda for better compatibility
conda create -n rag-env python=3.9
conda activate rag-env
conda install -c conda-forge faiss-cpu
pip install streamlit langchain-groq
```

### Linux

**Missing System Dependencies:**
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Components Individually

**Test PDF Loading:**
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("your_file.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")
```

**Test Embeddings:**
```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
test_embedding = embeddings.embed_query("test query")
print(f"Embedding created successfully: {len(test_embedding)} dimensions")
```

**Test Groq Connection:**
```python
from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key="your-key", model_name="llama3-8b-8192")
response = llm.invoke("Hello, world!")
print(response.content)
```

## Getting Help

If none of these solutions work:

1. **Check GitHub Issues**: Look for similar problems in the repository
2. **Create Detailed Issue**: Include:
   - Error message (full traceback)
   - Operating system and Python version
   - Package versions (`pip list`)
   - Steps to reproduce
3. **Check Dependencies**: Ensure all packages are compatible versions
4. **Update Everything**: Sometimes a simple update fixes issues

## Useful Commands

**Check Python Environment:**
```bash
python --version
pip list | grep -E "(streamlit|langchain|torch|faiss)"
```

**Reset Streamlit Cache:**
```bash
streamlit cache clear
```

**Check Disk Space:**
```bash
df -h  # Linux/Mac
dir   # Windows
```

**Monitor Memory Usage:**
```bash
htop  # Linux
Activity Monitor  # Mac
Task Manager  # Windows
```