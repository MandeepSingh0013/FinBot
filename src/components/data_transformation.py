import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it up.")

# Function to extract text from multiple PDF files
def extract_text_from_pdfs(pdf_files):
    """Extracts text from multiple PDF files."""
    combined_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            combined_text += page.extract_text() or ""
    return combined_text

# Function to preprocess text by splitting into chunks
def preprocess_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings and store in ChromaDB
def save_to_chromadb(text_chunks, persist_directory="chroma_storage"):
    """Generates embeddings and saves to ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Using OpenAI for embeddings
    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

# Main script
def main(pdf_paths):
    """Main pipeline to process PDFs and save embeddings."""
    if not pdf_paths:
        raise ValueError("No PDF files provided. Please provide valid file paths.")

    # Step 1: Extract text from PDFs
    print("Extracting text from PDFs...")
    raw_text = extract_text_from_pdfs(pdf_paths)

    # Step 2: Preprocess text
    print("Splitting text into chunks...")
    text_chunks = preprocess_text(raw_text)

    # Step 3: Generate embeddings and save to ChromaDB
    print("Generating embeddings and saving to ChromaDB...")
    vector_store = save_to_chromadb(text_chunks)
    
    print(f"Data successfully saved to ChromaDB at {vector_store.persist_directory}")

# Usage
if __name__ == "__main__":
    # Replace with paths to your PDF files
    pdf_file_paths = ['./Data/Aadhaar_Act_2016_as_amended.pdf']
    print(pdf_file_paths)
    
    # Ensure the PDF files exist
    for path in pdf_file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    main(pdf_file_paths)
