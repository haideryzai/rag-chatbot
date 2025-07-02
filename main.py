import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.document_processor import process_document
from services.rag_pipeline import query_rag
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a text-based document.
    Supports .txt and .md files; .pdf is optional with pdfminer.
    """
    allowed_extensions = {".txt", ".md", ".pdf"}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    try:
        content = await file.read()
        if file_extension == ".pdf":
            from pdfminer.high_level import extract_text
            from io import BytesIO
            content = extract_text(BytesIO(content))
        else:
            content = content.decode("utf-8")
        
        # Process and index the document
        process_document(content, file.filename)
        return {"message": f"Document {file.filename} uploaded and indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query(request: QueryRequest):
    """
    Endpoint to handle user queries and return RAG-based answers.
    """
    try:
        response = query_rag(request.question)
        return {"answer": response["answer"], "sources": response["sources"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "healthy"}