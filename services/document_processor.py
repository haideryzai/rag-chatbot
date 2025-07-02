from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vector_store import store_embeddings

def process_document(content: str, filename: str):
    """
    Process and index a document by splitting it into chunks and storing embeddings.
    """
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    
    # Create metadata for each chunk
    metadatas = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
    
    # Store embeddings in FAISS
    store_embeddings(chunks, metadatas)