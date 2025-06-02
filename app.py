import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Set Streamlit page config
st.set_page_config(page_title="NSU SEPS Information Chatbot", layout="wide")

# Load environment variables
load_dotenv()

# Directory for PDF storage
PDF_STORAGE_DIR = r"C:\Users\User\Desktop\Chatbot"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Convert text to vectorstore using FAISS
def get_vectorstore(text_chunks):
    try:
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        return FAISS.from_documents(
            documents, SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        )
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Setup conversation chain with Mistral model via Ollama
def get_conversation_chain(vectorstore, model_name="mistral"):
    try:
        llm = Ollama(model=model_name)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

# Handle user input
def handle_userinput(user_question):
    try:
        if 'conversation' not in st.session_state or not st.session_state.conversation:
            st.warning("⚠️ Please upload and process a PDF first.")
            return None

        response = st.session_state.conversation.invoke({"question": user_question})  
        return response["answer"]
    except Exception as e:
        st.error(f"Error processing request: {e}")
        return None

# Load PDFs from folder
def load_saved_pdfs():
    return [os.path.join(PDF_STORAGE_DIR, f) for f in os.listdir(PDF_STORAGE_DIR) if f.endswith(".pdf")]

# Process PDF documents
def process_pdfs(pdf_files):
    try:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            conversation_chain = get_conversation_chain(vectorstore)
            st.session_state.conversation = conversation_chain
            st.success("✅ PDFs processed successfully!")
        else:
            st.error("❌ Error: Could not create vector store.")
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")

# Main Streamlit app
def main():
    st.title("NSU SEPS Information Chatbot")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write("💬 Ask Questions")
    user_question = st.chat_input("Enter your question...")

    if user_question:
        response = handle_userinput(user_question)
        if response:
            st.session_state.chat_history.append({"question": user_question, "answer": response})

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    saved_pdfs = load_saved_pdfs()
    with st.sidebar:
        st.subheader("📂 Manage Documents")
        if saved_pdfs:
            if st.button("🔄 Process Saved PDFs"):
                with st.spinner("Processing saved PDFs..."):
                    process_pdfs(saved_pdfs)
        else:
            st.warning("⚠️ No saved PDFs found. Please upload files.")

if __name__ == "__main__":
    main()
