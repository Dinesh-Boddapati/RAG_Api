import uvicorn
import os
import tempfile
import numpy as np
import faiss
import google.generativeai as genai
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import time # NEW: To time the process

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
except AttributeError as e:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the environment variable and restart the application.")
except Exception as e:
    print(f"An error occurred during genai configuration: {e}")

# --- Global Variables ---
chunks = []
index = None
dimension = 0

# --- Helper Functions (Unchanged) ---
def read_pdf(pdf_path: str) -> str:
    """Reads a PDF and returns its text content."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Splits text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(text: str) -> list[float]:
    """Generates an embedding for a given text chunk."""
    try:
        embed = genai.embed_content(model="models/embedding-001", content=text)
        return embed['embedding']
    except Exception as e:
        print(f"Error getting embedding for text: {text[:20]}... Error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding API error: {e}")

#
# --- THIS IS THE MODIFIED FUNCTION ---
#
def embed_chunks_batch(chunks: list[str]) -> np.ndarray:
    """Generates embeddings for a list of text chunks in batches."""
    batch_size = 100  # Google's API limit can be up to 100 per request
    all_embeddings = []
    
    print(f"Total chunks to embed: {len(chunks)}")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # NEW: Call embed_content with the entire batch list
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=batch  # Pass the list of strings
            )
            all_embeddings.extend(response['embedding'])
            print(f"Embedded batch {i//batch_size + 1}/{(len(chunks)//batch_size) + 1}")
        
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Handle error (e.g., retry or skip)
            # For simplicity, we'll just re-raise
            raise HTTPException(status_code=500, detail=f"Embedding API error: {str(e)}")
            
        # Optional: Add a small delay to respect rate limits if needed
        # time.sleep(1) 

    return np.array(all_embeddings).astype("float32")

# --- FastAPI App ---
app = FastAPI(
    title="Retrieval-Augmented Generation (RAG) API",
    description="An API for uploading documents and asking questions based on their content.",
    version="0.1.0",
)

class QueryRequest(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF, processes it, and creates a vector index.
    """
    global chunks, index, dimension
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    start_time = time.time() # NEW: Start timer

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"File saved temporarily to: {temp_file_path}")

        # 1. Read PDF
        print("Reading PDF...")
        text = read_pdf(temp_file_path)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # 2. Chunk text
        print("Chunking text...")
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create text chunks.")
        
        # 3. Embed chunks (MODIFIED)
        print("Embedding chunks in batches (this may still take a moment)...")
        embeddings = embed_chunks_batch(chunks) # MODIFIED: Call new batch function
        dimension = embeddings.shape[1]
        
        # 4. Create and add to FAISS index
        print("Creating FAISS index...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        end_time = time.time() # NEW: End timer
        processing_time = end_time - start_time

        print(f"Upload and processing complete in {processing_time:.2f} seconds.")
        return {
            "filename": file.filename,
            "status": "processing complete",
            "total_chunks": len(chunks),
            "embedding_dimension": dimension,
            "processing_time_seconds": round(processing_time, 2)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temp file: {temp_file_path}")


# --- (The rest of the file is unchanged) ---

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    """
    Asks a question against the indexed documents.
    """
    global index, chunks
    
    query = request.query
    print(f"Received query: {query}")

    if index is None:
        raise HTTPException(status_code=400, detail="No document has been uploaded yet. Please use the /upload endpoint first.")
    
    try:
        # 1. Embed the query
        print("Embedding query...")
        query_vector = np.array([get_embedding(query)]).astype("float32")

        # 2. Search the FAISS index
        print("Searching index...")
        k = 3
        D, I = index.search(query_vector, k)
        
        # 3. Retrieve the context chunks
        context_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(context_chunks)

        # 4. Build the prompt
        prompt = f"Use this context to answer:\n\n{context}\n\nQuestion:{query}\nAnswer:"
        print(f"Building prompt...")

        # 5. Generate the answer
        print("Generating answer...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        print("Answer generation complete.")
        return {
            "query": query,
            "answer": response.text,
            "context_used": context_chunks
        }
        
    except Exception as e:
        print(f"An error occurred while answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API. Go to /docs to see the endpoints."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)