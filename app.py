import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load API Key from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

if not openai_api_key:
    st.error("❌ Missing OpenAI API Key! Set OPENAI_API_KEY in your .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📄 RAG Assistant with LangChain")
st.write("Upload a PDF, and ask questions based on its content.")

# File upload
uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

if uploaded_file:
    st.write("🔍 Processing file...")

    # Save file temporarily
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Create FAISS vector store dynamically
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vector_store.as_retriever()

    st.success("✅ PDF processed successfully!")

    # Initialize Chat Model
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    # Set up memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create RAG Chain
    rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.write("💬 Ask questions about the PDF below:")

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get response from RAG model
        with st.spinner("Thinking..."):
            response = rag_chain.run(user_input)

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

    st.write("🤝 Thank you for using the RAG Assistant!")

    # Cleanup: Remove temp file
    os.remove(file_path)
