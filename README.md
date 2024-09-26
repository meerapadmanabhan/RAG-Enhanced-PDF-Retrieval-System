# 📚 RAG-Enhanced PDF Retrieval System ✨
![app_page-0001](https://github.com/user-attachments/assets/144fedef-e284-431e-b310-ec7d0421a3be)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF files, extract text, and ask questions related to the content of the documents. The system leverages Pinecone for efficient retrieval, HuggingFace for embeddings, and BM25 for sparse encoding of textual data.

## Live Demo
Check out the live demo of the application here: [RAG PDF Retrieval System](https://rag-enhanced-pdf-retrieval-system.streamlit.app/)


## Features
- **PDF Upload**: Users can upload PDF documents for analysis.
- **Text Extraction**: The system extracts text from uploaded PDFs.
- **Hybrid Search**: Combines dense embeddings and sparse encoding for better retrieval results.
- **Interactive Query Interface**: Users can ask questions and receive relevant answers based on the PDF content.

## Technologies Used
- [Streamlit](https://streamlit.io/) for the web application framework.
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for extracting text from PDF files.
- [Pinecone](https://www.pinecone.io/) for vector database and indexing.
- [Hugging Face Transformers](https://huggingface.co/) for embeddings.
- [NLTK](https://www.nltk.org/) for natural language processing tasks.
- [LangChain](https://langchain.com/) for managing and processing language models.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/meerapadmanabhan/RAG-Enhanced-PDF-Retrieval-System.git
   cd rag-pdf-retrieval
2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

3. **Set up your Pinecone account and create an API key.**
    You can store your API key in a .env file or directly in your environment variables:

```bash
PINECONE_API_KEY=<your_api_key>
```

4. **Run the Streamlit application:**

```bash
streamlit run app.py
```

## Conclusion

The RAG-Enhanced PDF Retrieval System efficiently retrieves relevant information from uploaded PDF documents based on user queries. 

### Model Outputs
When a user submits a question, the system utilizes a hybrid search mechanism combining dense embeddings and sparse encoding to fetch the most pertinent sentences from the extracted text. The outputs include:

- **Relevant Sentences**: A list of sentences from the PDF that best match the user's query.
- **Scores**: Each output is accompanied by a score indicating its relevance, calculated based on the similarity between the query and the retrieved sentences. A higher score signifies a closer match, providing users with confidence in the information retrieved.

### Future Updates
I am continuously working to improve the system. Upcoming updates will focus on enhancing the model's ability to generate explanation sentences in a more natural and detailed manner, allowing for a more conversational and user-friendly experience. 

