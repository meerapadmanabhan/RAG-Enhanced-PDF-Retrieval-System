import streamlit as st
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
import nltk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download necessary NLTK data
nltk.download('punkt')

# Constants
INDEX_NAME = "infogenie"
EMBEDDING_MODEL = "all-MiniLM-L6-V2"

# Initialize Pinecone
def initialize_pinecone():
    try:
        api_key = st.secrets["PINECONE"]["PINECONE_API_KEY"]
        pc = Pinecone(api_key=api_key)
        return pc
    except KeyError as e:
        logging.error(f"KeyError: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {str(e)}")
        return None

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF file: {str(e)}")
        return ""

# Main logic
pc = initialize_pinecone()
if pc:
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(name=INDEX_NAME, dimension=384, metric="dotproduct", 
                             spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        logging.error(f"Error creating/loading index: {str(e)}")
        index = None

    # Continue with embedding and retriever setup...
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    bm25_encoder = BM25Encoder().default()

    # Streamlit UI
    st.title("📚 RAG-Enhanced PDF Retrieval System ✨")
    st.image("data/robot.jpg", caption="Welcome to the RAG System!", use_column_width=True)

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        extracted_text = extract_text_from_pdf(uploaded_file)
        sentences = nltk.sent_tokenize(extracted_text)  # More robust sentence splitting

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            # Fit BM25 encoder with the extracted sentences
            if len(sentences) > 1:
                bm25_encoder.fit(sentences)
                retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
                retriever.add_texts(sentences)
                logging.info("PDF text successfully indexed.")
            else:
                logging.warning("Not enough sentences to index.")

        query = st.text_input("Ask a question about the content in the PDF:")
        if st.button("Search"):
            if query:
                results = retriever.invoke(query)
                if results:
                    st.write(f"Results for: **{query}**")
                    contents = [result.page_content for result in results]
                    st.write(" ".join(contents))
                else:
                    st.write("No results found!")
            else:
                st.write("Please enter a query to search!")
else:
    logging.error("Pinecone not initialized.")
