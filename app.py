## Version 4 - To check particular similarity, removed negative similarity.

import os
import streamlit as st
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

nltk.download('punkt')

os.environ["GROQ_API_KEY"] = "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"

def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

def load_hidden_pdfs(directory="db_docs"):
    all_texts = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])
            filenames.extend([filename] * len(pages))  
    return all_texts, filenames

def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# New function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_text(text)

# Function to split text into sentences
def split_text_into_sentences(text):
    return sent_tokenize(text)

st.title("RAG Similarity App")
st.write("Please upload your PDF document to get similarity results.")

DB_DIRECTORY = "db_docs"

if not os.path.exists(DB_DIRECTORY):
    os.makedirs(DB_DIRECTORY)

st.info("Loading and processing database documents...")
db_texts, db_filenames = load_hidden_pdfs(directory=DB_DIRECTORY)
vector_store = create_vector_store(db_texts)
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    st.info("Processing uploaded file...")
    user_text = extract_text_from_pdf(uploaded_file)
    user_sentences = split_text_into_sentences(user_text)  # Split uploaded file into sentences
    
    st.info("Running similarity search...")
    results = []

    # Search each sentence from the uploaded document against the vector store
    for sentence in user_sentences:
        similarity_results = vector_store.similarity_search_with_score(sentence)
        results.extend(similarity_results)
    
    st.subheader("Similarity Results")

    for i, (doc, score) in enumerate(results):
        if score < 0:  # Skip if the similarity score is negative
            continue
        
        # Calculate similarity percentage (if needed, just for reference, can be removed later)
        similarity_percentage = round((1 - score) * 100, 2)
        document_name = db_filenames[i]

        # Show document name and similarity percentage
        st.write(f"Document: {document_name} - Similarity Score: {similarity_percentage}%")
        
        # Get sentences from the database document and the uploaded document
        db_sentences = split_text_into_sentences(doc.page_content)  # Split the DB document into sentences
        
        # Calculate how many similar sentences to show based on similarity score
        num_similar_sentences = max(1, int(len(db_sentences) * (1 - score)))  # More similar sentences for higher score
        
        # Get the most similar sentences
        st.write("Similar Sentences from the Database Document:")
        for j in range(num_similar_sentences):
            st.write(f"- {db_sentences[j]}")  # Display similar sentences from the database
        
        st.write("Similar Sentence from the Uploaded Document:")
        st.write(f"- {user_sentences[i % len(user_sentences)]}")  # Show the corresponding sentence from the uploaded document
        
    st.success("Ready to process more documents!")


## Version 3 - New code to see what is similarity 

# import os
# import streamlit as st
# from pypdf import PdfReader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter

# os.environ["GROQ_API_KEY"] = "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"

# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# def load_hidden_pdfs(directory="db_docs"):
#     all_texts = []
#     filenames = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#             filenames.extend([filename] * len(pages))  
#     return all_texts, filenames

# def create_vector_store(document_texts):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(document_texts, embedder)

# def extract_text_from_pdf(pdf_file):
#     reader = PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # New function to split text into chunks
# def split_text_into_chunks(text, chunk_size=1000):
#     text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
#     return text_splitter.split_text(text)

# st.title("RAG Similarity App")
# st.write("Please upload your PDF document to get similarity percentage.")

# DB_DIRECTORY = "db_docs"

# if not os.path.exists(DB_DIRECTORY):
#     os.makedirs(DB_DIRECTORY)

# st.info("Loading and processing database documents...")
# db_texts, db_filenames = load_hidden_pdfs(directory=DB_DIRECTORY)
# vector_store = create_vector_store(db_texts)
# uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# if uploaded_file is not None:
#     st.info("Processing uploaded file...")
#     user_text = extract_text_from_pdf(uploaded_file)
#     user_chunks = split_text_into_chunks(user_text)
    
#     st.info("Running similarity search...")
#     results = []

#     # Search each chunk from the uploaded document against the vector store
#     for chunk in user_chunks:
#         similarity_results = vector_store.similarity_search_with_score(chunk)
#         results.extend(similarity_results)
    
#     st.subheader("Similarity Results")

#     for i, (doc, score) in enumerate(results):
#         # Calculate similarity percentage
#         similarity_percentage = round((1 - score) * 100, 2)
#         document_name = db_filenames[i]
        
#         # Show document name and similarity percentage
#         st.write(f"Document: {document_name} - Similarity: {similarity_percentage}%")
        
#         # Display the similar lines (segments) from the uploaded document and the database document
#         db_segment = doc.page_content[:500]  # Accessing the page_content of doc and slicing
#         uploaded_segment = user_chunks[i % len(user_chunks)][:500]  # Accessing a chunk from the uploaded document

#         st.write(f"Database Document Segment: {db_segment}")
#         st.write(f"Uploaded Document Segment: {uploaded_segment}")
        
#     st.success("Ready to process more documents!")



## Version 2 - New Code to tell the document name as well. 

# import os
# import streamlit as st
# from pypdf import PdfReader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader

# os.environ["GROQ_API_KEY"] = "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"

# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# def load_hidden_pdfs(directory="db_docs"):
#     all_texts = []
#     filenames = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#             filenames.extend([filename] * len(pages))  
#     return all_texts, filenames

# def create_vector_store(document_texts):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(document_texts, embedder)

# def extract_text_from_pdf(pdf_file):
#     reader = PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# st.title("RAG Similarity App")
# st.write("Please upload your PDF document to get similarity percentage.")

# DB_DIRECTORY = "db_docs"

# if not os.path.exists(DB_DIRECTORY):
#     os.makedirs(DB_DIRECTORY)

# st.info("Loading and processing database documents...")
# db_texts, db_filenames = load_hidden_pdfs(directory=DB_DIRECTORY)
# vector_store = create_vector_store(db_texts)
# uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# if uploaded_file is not None:
#     st.info("Processing uploaded file...")
#     user_text = extract_text_from_pdf(uploaded_file)
#     st.info("Running similarity search...")
#     results = vector_store.similarity_search_with_score(user_text)
    
#     st.subheader("Similarity Results")
#     for i, (doc, score) in enumerate(results):
#         similarity_percentage = round((1 - score) * 100, 2)  
#         document_name = db_filenames[i]  
#         st.write(f"Document: {document_name} - Similarity: {similarity_percentage}%")

# st.success("Ready to process more documents!")


## Version 1

# import os
# import streamlit as st
# from pypdf import PdfReader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader

# os.environ["GROQ_API_KEY"] = "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"


# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# def load_hidden_pdfs(directory="db_docs"):
#     all_texts = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#     return all_texts

# def create_vector_store(document_texts):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(document_texts, embedder)

# def extract_text_from_pdf(pdf_file):
#     reader = PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# st.title("RAG Similarity App")
# st.write("Please upload your PDF document to get similarity percentage.")

# DB_DIRECTORY = "db_docs"

# if not os.path.exists(DB_DIRECTORY):
#     os.makedirs(DB_DIRECTORY)

# st.info("Loading and processing database documents...")
# db_texts = load_hidden_pdfs(directory=DB_DIRECTORY)
# vector_store = create_vector_store(db_texts)
# uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# if uploaded_file is not None:
#     st.info("Processing uploaded file...")
#     user_text = extract_text_from_pdf(uploaded_file)
#     st.info("Running similarity search...")
#     results = vector_store.similarity_search_with_score(user_text)
#     st.subheader("Similarity Results")
#     for i, (doc, score) in enumerate(results):
#         similarity_percentage = round((1 - score) * 100, 2)  
#         st.write(f"Document {i + 1}: {similarity_percentage}% similarity")

# st.success("Ready to process more documents!")
