import os
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from docx import Document  

os.environ["GROQ_API_KEY"] = "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"

def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

def load_hidden_documents(directory="db_docs"):
    """
    This function loads both PDF and DOCX documents from the directory.
    """
    all_texts = []
    filenames = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])
            filenames.extend([filename] * len(pages))
        
        elif filename.endswith(".docx"):
            all_texts_from_docx, filenames_from_docx = extract_text_from_docx(file_path)
            all_texts.extend(all_texts_from_docx)
            filenames.extend(filenames_from_docx)
    
    return all_texts, filenames

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a given PDF file.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """
    Extracts text from a DOCX file.
    """
    doc = Document(docx_file)
    all_text = []
    filenames = []
    
    doc_text = "\n".join([para.text for para in doc.paragraphs])
    all_text.append(doc_text)
    
    filenames.append(os.path.basename(docx_file))
    
    return all_text, filenames

def create_vector_store(document_texts):
    """
    Create a vector store from the document texts using HuggingFaceEmbeddings.
    """
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

st.title("RAG Similarity App")
st.write("Please upload your PDF or DOCX document to get similarity percentage.")

DB_DIRECTORY = "db_docs"

if not os.path.exists(DB_DIRECTORY):
    os.makedirs(DB_DIRECTORY)

st.info("Loading and processing database documents...")
db_texts, db_filenames = load_hidden_documents(directory=DB_DIRECTORY)
vector_store = create_vector_store(db_texts)

uploaded_file = st.file_uploader("Upload a PDF or DOCX document", type=["pdf", "docx"])

if uploaded_file is not None:
    st.info("Processing uploaded file...")

    if uploaded_file.type == "application/pdf":
        user_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        user_text = "\n".join(extract_text_from_docx(uploaded_file)[0])  
    
    st.info("Running similarity search...")
    results = vector_store.similarity_search_with_score(user_text)

    st.subheader("Similarity Results")
    found_similarity = False

    for i, (doc, score) in enumerate(results):
        similarity_percentage = round((1 - score) * 100, 2)
        if similarity_percentage > 0:
            found_similarity = True
            document_name = db_filenames[i]
            st.write(f"Document: {document_name} - Similarity: {similarity_percentage:.2f}%")
    
    if not found_similarity:
        st.write("No similarity found.")

st.success("Ready to process more documents!")