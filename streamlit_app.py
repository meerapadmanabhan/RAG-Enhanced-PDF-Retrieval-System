import streamlit as st
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
import nltk
from collections.abc import Iterable  # Importing Iterable here


nltk.download('punkt')
nltk.download('punkt_tab')  # Add this line

# Debugging: Print available secrets
st.write(st.secrets)

# Initialize Pinecone
try:
    api_key = st.secrets["PINECONE"]["PINECONE_API_KEY"]
except KeyError as e:
    st.error(f"KeyError: {str(e)}. Check your secrets.toml file.")
    api_key = None  # Set api_key to None to prevent further errors

if api_key:
    pc = Pinecone(api_key=api_key)  # Create an instance of Pinecone


# Initialize Pinecone
#api_key = st.secrets["PINECONE"]["PINECONE_API_KEY"]  # Store Pinecone API key in secrets.toml
#pc = Pinecone(api_key=api_key)  # Create an instance of Pinecone

index_name = "infogenie"

# Create or load index
try:
    if index_name not in pc.list_indexes().names():  # Use the Pinecone instance to list indexes
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
except Exception as e:
    st.error(f"Error creating index: {str(e)}")

index = pc.Index(index_name)  # Use the Pinecone instance to get the index

# Generate embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
bm25_encoder = BM25Encoder().default()

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Read from the uploaded file
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Streamlit UI
st.title("📚 RAG-Enhanced PDF Retrieval System ✨")

# Display an image
st.image("/workspaces/RAG-Enhanced-PDF-Retrieval-System/data/robot.jpg", caption="Welcome to the RAG System!", use_column_width=True)

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    # Split text into sentences for indexing
    sentences = extracted_text.split(". ")  # Simple sentence splitting

    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]  # Ensure sentences are not empty or whitespace

    # Check if there are valid sentences to encode
    if sentences:
        # Fit BM25 encoder with the extracted sentences
        bm25_encoder.fit(sentences)

        # Initialize the retriever
        retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

        # Add sentences to the Pinecone index
        try:
            retriever.add_texts(sentences)  # Attempt to add texts to Pinecone
            st.success("PDF text successfully indexed. You can now search within the PDF.")
        except Exception as e:
            # Suppress the error message
            print(f"Error adding texts to Pinecone: {str(e)}")  # Log the error to the console instead
    else:
        st.warning("No valid sentences to index.")

    # Provide a query interface for the user
    query = st.text_input("Ask a question about the content in the PDF:", "")

    if st.button("Search"):
        if query:
            results = retriever.invoke(query)
            if results:
                st.write(f"Results for: **{query}**")
            
                # Create a list to hold the page contents
                contents = []
                for result in results:
                    contents.append(result.page_content)  # Correctly accessing page_content

                # Join all the contents into a single string
                combined_content = " ".join(contents)  # Combine into a single paragraph
                st.write(combined_content)  # Display the combined content
            
            else:
                st.warning("No results found!")
        else:
            st.warning("Please enter a query to search!")
