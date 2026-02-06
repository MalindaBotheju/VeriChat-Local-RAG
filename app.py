import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. App Config
st.set_page_config(page_title="VeriChat (Local Privacy)", page_icon="🛡️")
st.title("🛡️ VeriChat: Private AI Document Assistant")
st.caption("Running locally with Llama 3 | No data leaves this computer")

# 2. Setup the "Brain" (Ollama) & "Embeddings" (HuggingFace)
# We use caching so it doesn't reload every time you click a button
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatOllama(model="llama3")

embeddings = load_embedding_model()
llm = load_llm()

# 3. Session State (To remember the chat history)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# 4. Sidebar: Upload PDF
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload a Confidential PDF", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Encrypting & Indexing..."):
            # Save PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load and Split
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # --- ADD THIS CHECK HERE ---
            if len(splits) == 0:
                st.error("⚠️ No text found in this PDF! It might be a scanned image.")
                st.stop()
            # ---------------------------

            # Store in Vector DB (Chroma)
            st.session_state.vector_db = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            os.remove(tmp_path) # Clean up temp file
            st.success(f"Processed {len(splits)} text chunks successfully!")

# 5. The Chat Interface
# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("Ask a question about the document...")

if user_query:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Answer
    with st.chat_message("assistant"):
        if st.session_state.vector_db is None:
            st.error("Please upload and process a PDF first!")
        else:
            # RAG Pipeline
            retriever = st.session_state.vector_db.as_retriever()
            
            # The Prompt Template
            template = """Answer the question based ONLY on the following context:
            {context}
            
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(user_query)
            st.markdown(response)
            
            # Save context for history
            st.session_state.chat_history.append({"role": "assistant", "content": response})