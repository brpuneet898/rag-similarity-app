import os
import streamlit as st
# from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from groq.groq_lm import ChatGroq

os.environ["GROQ_API_KEY"] = "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"


def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

def load_hidden_pdfs(directory="db_docs"):
    all_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])
    return all_texts

def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("RAG Similarity App")
st.write("Please upload your PDF document to get similarity percentage.")

DB_DIRECTORY = "db_docs"

if not os.path.exists(DB_DIRECTORY):
    os.makedirs(DB_DIRECTORY)

st.info("Loading and processing database documents...")
db_texts = load_hidden_pdfs(directory=DB_DIRECTORY)
vector_store = create_vector_store(db_texts)
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    st.info("Processing uploaded file...")
    user_text = extract_text_from_pdf(uploaded_file)
    st.info("Running similarity search...")
    results = vector_store.similarity_search_with_score(user_text)
    st.subheader("Similarity Results")
    for i, (doc, score) in enumerate(results):
        similarity_percentage = round((1 - score) * 100, 2)  
        st.write(f"Document {i + 1}: {similarity_percentage}% similarity")

st.success("Ready to process more documents!")
