import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import tiktoken
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

def read_pdfs_from_folder(folder_path):
    """
    Reads all PDFs in the specified folder and extracts their text.

    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        dict: A dictionary with file names as keys and extracted text as values.
    """
    pdf_texts = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            pdf_reader = PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            pdf_texts[file_name] = text
    return pdf_texts

def clean_text_advanced(text):
    """
    Cleans and preprocesses text by removing noise patterns specific to government documents.
    Also handles artifacts like underscores, partial masking, and formatting issues.

    Args:
        text (str): Raw text extracted from PDF.

    Returns:
        str: Cleaned text.
    """
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Replace multiple spaces and line breaks with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    # Normalize text to lowercase
    text = text.lower()

    # Remove underscores and lines like "______ _"
    text = re.sub(r'_+', ' ', text)

    # Remove partially masked numbers or IDs like "2***" or similar patterns
    text = re.sub(r'\b\d\*+\b', '', text)

    # Remove common noise patterns
    text = re.sub(r'page \d+ of \d+', '', text)  # "Page X of Y"
    text = re.sub(r'page \d+', '', text)         # "Page X"
    text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', text)  # Timestamps
    text = re.sub(r'(confidential|proprietary|government seal|draft)', '', text, flags=re.IGNORECASE)

    # Custom patterns for government documents
    # Example: Remove standard disclaimers or legal boilerplate
    text = re.sub(r'this document is for informational purposes only.', '', text, flags=re.IGNORECASE)

    # Final clean-up of extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text

def split_text_into_chunks(text):
    """
    Splits text into smaller chunks suitable for embeddings.

    Args:
        text (str): Cleaned text.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

import chromadb
from chromadb.config import Settings
# from langchain.embeddings import OpenAIEmbeddings

def create_embeddings_and_save_to_chroma(text_chunks, file_name):
    """
    Creates embeddings for text chunks and saves them into ChromaDB.

    Args:
        text_chunks (list): List of text chunks.
        file_name (str): Name of the source file for metadata.

    Returns:
        None
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Initialize ChromaDB Client
    persistent_client  = chromadb.PersistentClient('vetorstore')

    
    # Create or retrieve the collection
    collection = persistent_client.get_or_create_collection(name="findata")


    # Prepare embeddings and metadata
    for idx, chunk in enumerate(text_chunks):
        embedding = embeddings.embed_query(chunk)
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": file_name, "chunk_id": idx}],
            ids=[f"{file_name}_chunk_{idx}"],
        )

    print(f"Added {len(text_chunks)} documents to the ChromaDB collection ")


def process_pdfs_and_save_to_chroma(folder_path):
    """
    End-to-end pipeline to process PDFs, clean text, create embeddings, and save to ChromaDB.

    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        None
    """
    pdf_texts = read_pdfs_from_folder(folder_path)

    for file_name, raw_text in pdf_texts.items():
        print(f"Processing {file_name}...")
        
        # Clean the text
        cleaned_text = clean_text_advanced(raw_text)
        
        # Split the cleaned text into chunks
        text_chunks = split_text_into_chunks(cleaned_text)
        
        # Create embeddings and save to ChromaDB
        create_embeddings_and_save_to_chroma(text_chunks, file_name)
        
        print(f"Finished processing {file_name}.")

# Usage Example
if __name__ == "__main__":
    folder_path = "./src/components/Data"  # Replace with your folder path
    process_pdfs_and_save_to_chroma(folder_path)
