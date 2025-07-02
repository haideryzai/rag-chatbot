# RAG Chatbot Project

A full-stack chatbot using Retrieval-Augmented Generation (RAG) to answer questions based on user-uploaded documents, using free, open-source models.

## Features
- **Backend**: FastAPI with LangChain, SentenceTransformers, FAISS, and Hugging Face's `facebook/bart-large` model.
- **Frontend**: React with Tailwind CSS for a simple chat and file upload interface.
- **Document Processing**: Supports `.txt`, `.md`, and `.pdf` files; chunks text and stores embeddings in FAISS.
- **Query Processing**: Retrieves relevant document chunks and generates answers using RAG.
- **Bonus**: Displays source document filenames in responses.

## Prerequisites
- Python 3.9+
- Node.js 16+ (optional, only if using the frontend)
- At least 8GB RAM (16GB recommended) for running `facebook/bart-large`
- (Optional) GPU with CUDA for faster inference

## Backend Setup Instructions

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   - For GPU support, install CUDA-enabled PyTorch:
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     ```
4. (Optional) Copy `.env.example` to `.env` if you need environment-specific configurations:
   ```bash
   cp .env.example .env
   ```
5. Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   The backend will be available at `http://localhost:8000`.

## Calling the APIs

The backend provides three endpoints: `/health`, `/upload`, and `/query`. Below are instructions to call each using `curl`. You can also use tools like Postman by importing these `curl` commands or configuring equivalent HTTP requests.

### 1. Health Check (`/health`)
- **Purpose**: Verify the backend is running.
- **Method**: GET
- **URL**: `http://localhost:8000/health`
- **curl Command**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Expected Response**:
  ```json
  {"status": "healthy"}
  ```
- **Postman Setup**:
  - Create a GET request to `http://localhost:8000/health`.
  - Send the request and check for the `healthy` status.
  - Alternatively, import the `curl` command in Postman’s `Import` > `Raw Text` section.

### 2. Upload Document (`/upload`)
- **Purpose**: Upload a document (`.txt`, `.md`, or `.pdf`) to be indexed in the FAISS vector store.
- **Method**: POST
- **URL**: `http://localhost:8000/upload`
- **Body**: Form-data with a `file` field containing the document.
- **Example**:
  Create a file named `company_profile.txt` with:
  ```
  Our company, Acme Corp, provides software development, cloud consulting, and AI solutions.
  We specialize in web and mobile app development, DevOps, and machine learning model deployment.
  ```
- **curl Command**:
  ```bash
  curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/company_profile.txt"
  ```
  - Replace `/path/to/company_profile.txt` with the actual file path.
  - For a PDF, use a text-based PDF (e.g., `company_profile.pdf`):
    ```bash
    curl -X POST http://localhost:8000/upload \
    -F "file=@/path/to/company_profile.pdf"
    ```
- **Expected Response**:
  ```json
  {"message": "Document company_profile.txt uploaded and indexed successfully"}
  ```
- **Postman Setup**:
  - Create a POST request to `http://localhost:8000/upload`.
  - In the `Body` tab, select `form-data`.
  - Add a key `file`, set type to `File`, and select your file (e.g., `company_profile.txt`).
  - Send the request.
  - Alternatively, import the `curl` command in Postman.
- **Notes**:
  - Only `.txt`, `.md`, and text-based `.pdf` files are supported.
  - If you get `Unsupported file type`, check the file extension.
  - The document is chunked and indexed in FAISS for querying.

### 3. Query (`/query`)
- **Purpose**: Ask a question and receive an answer based on indexed documents.
- **Method**: POST
- **URL**: `http://localhost:8000/query`
- **Body**: JSON with a `question` field.
- **curl Command**:
  ```bash
  curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does the company provide?"}'
  ```
- **Expected Response** (assuming `company_profile.txt` was uploaded):
  ```json
  {
    "answer": "Acme Corp provides software development, cloud consulting, and AI solutions, specializing in web and mobile app development, DevOps, and machine learning model deployment.",
    "sources": [
      {"filename": "company_profile.txt", "chunk_id": 0}
    ]
  }
  ```
- **Postman Setup**:
  - Create a POST request to `http://localhost:8000/query`.
  - In the `Headers` tab, add `Content-Type: application/json`.
  - In the `Body` tab, select `raw`, choose `JSON`, and enter:
    ```json
    {"question": "What services does the company provide?"}
    ```
  - Send the request.
  - Alternatively, import the `curl` command in Postman.
- **Additional Example**:
  ```bash
  curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Acme Corp specialize in?"}'
  ```
  Expected response:
  ```json
  {
    "answer": "Acme Corp specializes in web and mobile app development, DevOps, and machine learning model deployment.",
    "sources": [
      {"filename": "company_profile.txt", "chunk_id": 0}
    ]
  }
  ```
- **Notes**:
  - Ensure a document is uploaded before querying, or the answer may be empty.
  - The `facebook/bart-large` model may produce slightly varied answers compared to larger models.

## Troubleshooting
- **Server Not Running**:
  - If `curl` returns `Connection refused`, ensure the server is running (`uvicorn main:app --host 0.0.0.0 --port 8000`).
  - Check terminal output for errors.
- **Model Loading Issues**:
  - If the backend crashes, ensure you have enough RAM (8GB minimum, 16GB recommended).
  - Try a smaller model like `google/flan-t5-small` in `services/rag_pipeline.py`:
    ```python
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    ```
  - Ensure `transformers` and `torch` are installed.
- **Slow Inference**:
  - CPU inference may take seconds per query. For GPU support, set `device=0` in `services/rag_pipeline.py` and install CUDA-enabled `torch`.
- **PDF Issues**:
  - If PDF upload fails, ensure `pdfminer.six` is installed and the PDF is text-based.
- **Poor Answer Quality**:
  - If answers are vague, ensure documents are indexed. For better results, consider fine-tuning `facebook/bart-large` or using a larger model like `meta-llama/Llama-2-7b` (requires Hugging Face approval).

## Notes
- The FAISS index is stored in `backend/faiss_index`.
- To use a different model, update `services/rag_pipeline.py` with the desired Hugging Face model (e.g., `google/flan-t5-base`).
- For GPU support, ensure CUDA is installed and update `device=0` in `services/rag_pipeline.py`.