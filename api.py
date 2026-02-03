from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
from qdrant_client import QdrantClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct
import uuid
from config import get_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
UPLOAD_DIR = Path("data")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(str(file_path))
        elif file.filename.endswith('.txt'):
            loader = TextLoader(str(file_path))
        else:
            raise HTTPException(400, "only pdf and txt files supported")
        
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings_model = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url="http://localhost:11434"
        )
        
        client = QdrantClient(url=QDRANT_URL)
        points = []
        
        for chunk in chunks:
            embedding = embeddings_model.embed_query(chunk.page_content)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    "source": file.filename,
                    "page": chunk.metadata.get("page", 0)
                }
            )
            points.append(point)
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        return {
            "message": f"uploaded {file.filename}",
            "chunks": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        client = QdrantClient(url=QDRANT_URL)
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url="http://localhost:11434"
        )
        
        query_vector = embeddings.embed_query(request.question)
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=request.top_k
        )
        
        if not results:
            return QueryResponse(answer="no relevant documents found", sources=[])
        
        context = "\n\n".join([r.payload["text"] for r in results])
        
        llm = get_llm()
        
        template = """Use the following context to answer the question. If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        formatted_prompt = prompt.format(context=context, question=request.question)
        
        answer = llm.invoke(formatted_prompt)
        
        sources = [
            {
                "source": r.payload["source"],
                "score": float(r.score),
                "text": r.payload["text"][:200]
            }
            for r in results
        ]
        
        return QueryResponse(answer=answer, sources=sources)
    
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)