import os
from enum import Enum

class LLMProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_llm_provider():
    if ENVIRONMENT == "production" and OPENAI_API_KEY:
        return LLMProvider.OPENAI
    return LLMProvider.OLLAMA

def get_llm():
    provider = get_llm_provider()
    
    if provider == LLMProvider.OPENAI:
        from langchain_community.llms import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    else:
        from langchain_community.llms import Ollama
        return Ollama(model="llama3.2", base_url="http://localhost:11434")