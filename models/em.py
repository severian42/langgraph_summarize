import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

load_dotenv()

def get_gemini_embeddings(**kwargs):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

def get_openai_embeddings(**kwargs):
    return OpenAIEmbeddings()

def get_huggingface_embeddings(**kwargs):
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

def get_ollama_embeddings(**kwargs):
    return OllamaEmbeddings(model="llama2")

# Map of provider names to their initialization functions
EMBEDDING_PROVIDERS = {
    "gemini": get_gemini_embeddings,
    "openai": get_openai_embeddings,
    "huggingface": get_huggingface_embeddings,
    "ollama": get_ollama_embeddings
}

_current_embedding_model = None

def get_embedding_model(provider_name="gemini"):
    """
    Get or create an embedding model instance.
    
    Args:
        provider_name (str): Name of the embedding provider to use
        
    Returns:
        Embedding model instance
    """
    global _current_embedding_model
    
    if provider_name not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {provider_name}")
    
    # Create new embedding model instance with the specified provider
    _current_embedding_model = EMBEDDING_PROVIDERS[provider_name]()
    
    return _current_embedding_model

# For backwards compatibility, keep the default embedding instance
embedding = get_embedding_model("gemini")
