from langchain.prompts import PromptTemplate
from services.vector_store import search_similar
from transformers import pipeline

# Initialize Hugging Face model for text generation
generator = pipeline("text2text-generation", model="facebook/bart-large", device=-1)  # device=-1 for CPU, 0 for GPU

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="You are a helpful assistant. Based on the following context, answer the question concisely and accurately.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

def query_rag(question: str):
    """
    Process a user query using the RAG pipeline: retrieve relevant chunks and generate an answer.
    """
    # Retrieve relevant document chunks
    docs = search_similar(question, k=3)
    
    # Construct context from retrieved documents
    context = "\n".join([doc["content"] for doc in docs])
    sources = [doc["metadata"] for doc in docs]
    
    # Generate answer using Hugging Face model
    prompt = prompt_template.format(question=question, context=context)
    # Truncate prompt if too long for the model (BART has a max length of 1024 tokens)
    prompt = prompt[:1000]
    answer = generator(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
    
    return {"answer": answer.strip(), "sources": sources}