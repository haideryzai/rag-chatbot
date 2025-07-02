from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

import os

# Initialize embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize FAISS index (stored locally)
vector_store = None
FAISS_INDEX_PATH = "./faiss_index"

def store_embeddings(texts, metadatas):
    """
    Store text chunks and their embeddings in FAISS.
    """
    global vector_store
    if vector_store is None:
        # Create new FAISS index
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    else:
        # Add to existing FAISS index
        vector_store.add_texts(texts, metadatas=metadatas)
    
    # Save index to disk
    vector_store.save_local(FAISS_INDEX_PATH)

def load_vector_store():
    """
    Load FAISS index from disk if it exists.
    """
    global vector_store
    if os.path.exists(FAISS_INDEX_PATH) and vector_store is None:
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def search_similar(query: str, k: int = 3):
    """
    Search for top-k similar document chunks to the query.
    """
    load_vector_store()
    if vector_store is None:
        return []
    docs = vector_store.similarity_search(query, k=k)
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]