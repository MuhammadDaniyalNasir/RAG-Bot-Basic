import streamlit as st

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
import warnings
warnings.filterwarnings("ignore")

st.title(" RAG Chatbot - Ask Questions About Your PDF!")

st.sidebar.header("RAG Chatbot")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000 )
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200)
num_sources = st.sidebar.slider("Number of Sources to Retrieve", 1, 10, 5)

#File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type = "pdf")

#Info about the current document 
if os.path.exists ("reflexion.pdf"):
    st.sidebar.success("Default PDF 'reflexion.pdf' found")
    
elif uploaded_file:
    st.sidebar.sucess("PDF Uploaded Successfully")
    
else:
    st.sidebar.warning("⚠️ No PDF found. Please upload a PDF or add 'reflexion.pdf' to your directory")
    
# Setup session variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message["content"])

@st.cache_resource
def get_vectorstore(_uploaded_file = None, _chunk_size = 1000, _chunk_overlap = 200):
    try:
        # Determine which PDF to use
        if _uploaded_file is not None:
            #Save uploaded file temporarily 
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(_uploaded_file.getbuffer())
            pdf_name = "temp_uploaded.pdf"
        
        else:
            pdf_name = "reflexion.pdf"
            
        # Check if PDF exists 
        if not os.path.exists(pdf_name):
            st.error(f"PDF file '{pdf_name}' not found. PLease make sure the file exists in same directory. ")
            return None
        
        # Load the PDF
        loader = PyPDFLoader(pdf_name)
        documents = loader.load()
        
        if not documents:
            st.error("No content found in the PDF.")
            return None
        
        st.sidebar.info(f" Loaded {len(documents)} pages from PDF")
        
        # Split documents into chunks 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        st.sidebar.info(f" Created {len(chunks)} text chunks")
        
        # Create Embeddings
        embeddings = HuggingFaceBgeEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L12-v2"
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        st.sidebar.success(" vector store created succesfully!")
        
        return vectorstore
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None
    

# Chat Input
prompt = st.chat_input("Pass your Prompt")

if prompt:
    model = "llama3-8b-8192"
    
    # Add user message to the chat
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content' : prompt})
    
    # Initialize Groq Chat
    groq_chat = ChatGroq(
        groq_api_key = "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",
        model_name = model,
    )
    try:
        # Get vectorstore with current settings
        vectorstore = get_vectorstore(uploaded_file, chunk_size, chunk_overlap)
        
        if vectorstore is None:
            st.error("❌ Failed to load the document. Please check if your PDF exists or upload a new one.")
            
        else:
            # Create RAG prompt template
            rag_prompt = ChatPromptTemplate.from_template(
                """You are a helpful assistant that answers questions based on the provided document context.
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
                
                Answer:"""
                
            )
            
            # Create Retriever
            retriever = vectorstore.as_retriever(search_kwargs = {"k": num_sources})
            
            # Create RAG chain using LCEL (LangChain Expression Language)
            
            def format_docs(docs):
                return "\n\n".join(f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(docs))
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | groq_chat
                | StrOutputParser()
            )
            
            # Get response from RAG chain
            with st.spinner("🔍 Searching through the document..."):
               response = rag_chain.invoke(prompt)
            
            #Display response
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({'role':'assistant', 'content': response})
            
            # Show source information
            with st.expander("📚 View Source Documents"):
                retrieved_docs = retriever.get_relevant_documents(prompt)
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"**📄 Source {i+1}:**")
                    st.write(doc.page_content[: 800] + "..." if len(doc.page_content) > 800 else doc.page_content)

                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.write(f"**📝 Metadata:** {doc.metadata}")
                    st.write("---")
                    
            # Show similarity scores if available
            with st.expander("🎯 Relevance Scores"):
                docs_with_scores = vectorstore.similarity_search_with_score(prompt,
                            k = num_sources)

                for i, (doc, score) in enumerate(docs_with_scores):
                    st.write(f"**📄 Source {i+1}: - Similarity Score : {score:.3f}**")
                    st.write(doc.page_content[:300] + "...")
                    st.write("---")
                    
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        # Fallback to simple chat witout RAG
        try:
            simple_prompt = ChatPromptTemplate.from_template(
                """You are a helpful assistant. Answer the following question: {user_prompt}
                Start with a summary sentence, then provide detailed points.
                Talk like a cool guy using phrases like 'watch ya doin'.
                """
            )
            
            simple_chain = simple_prompt | groq_chat | StrOutputParser()
            response = simple_chain.invoke({
                "user_prompt": prompt
            })
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({
                'role': 'assistant', 'content': response
            })
            
        except Exception as fallback_error:
            st.error(f"Fallback error: {str(fallback_error)}")
                    
