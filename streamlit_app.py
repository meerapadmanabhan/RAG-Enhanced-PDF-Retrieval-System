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
nltk.download('punkt_tab')

# Initialize Pinecone
try:
    api_key = st.secrets["PINECONE"]["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)  # Create an instance of Pinecone
except KeyError as e:
    logging.error(f"KeyError: {str(e)}")  # Log error without showing it
    pc = None  # Set pc to None to prevent further errors
except Exception as e:
    logging.error(f"Failed to initialize Pinecone: {str(e)}")  # Log error without showing it
    pc = None  # Set pc to None to prevent further errors

# Only proceed if pc is defined
if pc:
    index_name = "infogenie"

    # Create or load index
    try:
        if index_name not in pc.list_indexes().names():  # Check if the index exists
            pc.create_index(
                name=index_name,
                dimension=384, 
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(index_name)  # Use the Pinecone instance to get the index
    except Exception as e:
        logging.error(f"Error creating/loading index: {str(e)}")  # Log error without showing it
        index = None  # Set index to None to prevent further errors

    # Generate embeddings and BM25 encoder
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
    bm25_encoder = BM25Encoder().default()

    # Function to extract text from a PDF file
    def extract_text_from_pdf(pdf_file):
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Read from the uploaded file
            text = ""
            for page in doc:
                text += page.get_text("text")
            return text
        except Exception as e:
            logging.error(f"Error reading PDF file: {str(e)}")  # Log error without showing it
            return ""  # Return empty string for error cases

    # Streamlit UI
    st.title("📚 RAG-Enhanced PDF Retrieval System ✨")

    # Display an image
    st.image("data/robot.jpg", caption="Welcome to the RAG System!", use_column_width=True)

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
            if len(sentences) > 1:  # Ensure there are enough sentences to fit
                bm25_encoder.fit(sentences)

                # Initialize the retriever only after adding texts to the Pinecone index
                try:
                    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
                    
                    # Attempt to add texts to Pinecone, ensuring they are valid
                    if any(len(sentence.split()) > 0 for sentence in sentences):  # Check if any sentence has valid content
                        retriever.add_texts(sentences)  # Add texts to Pinecone
                        logging.info("PDF text successfully indexed.")  # Log success without showing it
                    else:
                        logging.warning("No valid sentences to add to Pinecone.")  # Log warning without showing it
                except Exception as e:
                    logging.error(f"Error adding texts to Pinecone: {str(e)}")  # Log error without showing it
            else:
                logging.warning("Not enough sentences to index.")  # Log warning without showing it
        else:
            logging.warning("No valid sentences to index.")  # Log warning without showing it

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
                    st.write("No results found!")  # General message for no results
            else:
                st.write("Please enter a query to search!")  # General message for empty query
else:
    logging.error("Pinecone not initialized.")  # Log error without showing it
