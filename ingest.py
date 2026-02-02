from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from pathlib import Path
import uuid

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OLLAMA_MODEL = "nomic-embed-text"

def setup_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    
    return client

def load_documents(data_dir="data"):
    documents = []
    data_path = Path(data_dir)
    
    pdf_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()
    documents.extend(pdf_docs)
    
    txt_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    txt_docs = txt_loader.load()
    documents.extend(txt_docs)
    
    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    embeddings_model = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url="http://localhost:11434"
    )
    
    points = []
    
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk.page_content)
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0)
            }
        )
        points.append(point)
    
    return points

def store_in_qdrant(client, points):
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

def main():
    client = setup_qdrant()
    documents = load_documents()
    
    if not documents:
        print("no documents in data/ folder")
        return
    
    chunks = chunk_documents(documents)
    points = create_embeddings(chunks)
    store_in_qdrant(client, points)
    
    print("done")

if __name__ == "__main__":
    main()