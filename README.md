# RAG (Retrieval-Augmented Generation) API

This project is a complete, containerized Python backend that allows you to "chat with your documents." It uses a RAG pipeline to provide context-aware answers from an uploaded PDF file, all served via a clean FastAPI interface.

This project was built to demonstrate a full end-to-end AI/ML workflow, from local development to production cloud deployment.

### 🚀 Live Demo

The API is deployed on Google Cloud Run and is publicly accessible:
 https://rag-api-service-871488165179.us-central1.run.app/docs

---

## ✨ Features

* **PDF Upload:** A `POST /upload/` endpoint to upload any PDF.
* **Text Processing:** Automatically reads, chunks, and creates vector embeddings for the PDF content.
* **Vector Search:** Uses a FAISS index for high-speed similarity search to find relevant context.
* **Generative AI:** Leverages Google's `gemini-2.5-flash` model to generate accurate answers based *only* on the provided context.
* **Scalable:** Fully containerized with Docker and deployed as a serverless microservice.

---

## 🛠️ Tech Stack
<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://cloud.google.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/google_cloud/google_cloud-icon.svg" alt="gcp" width="40" height="40"/> </a> <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> </p>

* **Backend:** Python
* **API Framework:** FastAPI
* **Generative AI:** Google Generative AI (Gemini 2.5 Flash)
* **Embedding Model:** `models/embedding-001`
* **Vector Search:** FAISS (Facebook AI Similarity Search)
* **PDF Parsing:** PyPDF
* **Containerization:** Docker
* **Cloud Platform:** Google Cloud Run
---

## 🐳 How to Run Locally

You can run this entire application on your local machine using Docker.

### Prerequisites

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* A [Google Gemini API Key](https://ai.google.dev/gemini-api/docs/api-key).

### 1. Clone the Repository

bash
git clone [https://github.com/YourUsername/rag-api.git](https://github.com/YourUsername/rag-api.git)
cd rag-api

docker build -t rag-api .

docker run -p 8000:8000 -e GEMINI_API_KEY="YOUR_ACTUAL_API_KEY" rag-api

## Use the API
Your API is now running! Go to your browser to see the interactive documentation: http://127.0.0.1:8000/docs

📖 API Endpoints
You can test both endpoints directly from the /docs page.

1. POST /upload/
Description: Uploads a PDF file to be processed. This will read the text, create embeddings, and build the in-memory FAISS index. You must do this first.

Request Body: multipart/form-data (Just select a .pdf file).

Response:

JSON

{
  "filename": "your-file.pdf",
  "status": "processing complete",
  "total_chunks": 120,
  "embedding_dimension": 768,
  "processing_time_seconds": 5.42
}
2. POST /ask/
Description: Asks a question against the most recently uploaded PDF.

Request Body:

JSON

{
  "query": "What is this document about?"
}
Response:

JSON

{
  "query": "What is this document about?",
  "answer": "This document is a clear and simple guide to [....]",
  "context_used": [
    "Chunk 1 of text...",
    "Chunk 2 of text...",
    "Chunk 3 of text..."
  ]
}
