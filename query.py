from qdrant_client import QdrantClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3.2"

def search_documents(query, top_k=3):
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url="http://localhost:11434"
    )
    
    query_vector = embeddings.embed_query(query)
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    
    return results

def generate_answer(query, context):
    llm = Ollama(
        model=OLLAMA_LLM_MODEL,
        base_url="http://localhost:11434"
    )
    
    template = """Use the following context to answer the question. If you don't know, say so.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context=context, question=query)
    
    response = llm.invoke(formatted_prompt)
    return response

def rag_query(question):
    print(f"\nQuestion: {question}")
    print("\nSearching documents...")
    
    results = search_documents(question)
    
    if not results:
        return "No relevant documents found."
    
    print(f"Found {len(results)} relevant chunks\n")
    
    context = "\n\n".join([result.payload["text"] for result in results])
    
    print("Generating answer...")
    answer = generate_answer(question, context)
    
    print(f"\nAnswer: {answer}\n")
    
    print("Sources:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.payload['source']} (score: {result.score:.3f})")
    
    return answer

if __name__ == "__main__":
    question = "What does AWS provide?"
    rag_query(question)