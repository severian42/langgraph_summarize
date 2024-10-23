import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub, Ollama

load_dotenv()

def get_gemini_llm(**kwargs):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

def get_openai_llm(**kwargs):
    return ChatOpenAI(temperature=0)

def get_huggingface_llm(**kwargs):
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    return HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        huggingfacehub_api_token=huggingface_api_key
    )

def get_ollama_llm(**kwargs):
    return Ollama(model="llama2")

# Map of provider names to their initialization functions
LLM_PROVIDERS = {
    "gemini": get_gemini_llm,
    "openai": get_openai_llm,
    "huggingface": get_huggingface_llm,
    "ollama": get_ollama_llm
}

_current_llm = None

def get_llm(provider_name="gemini"):
    """
    Get or create an LLM instance.
    
    Args:
        provider_name (str): Name of the LLM provider to use
        
    Returns:
        LLM instance
    """
    global _current_llm
    
    if provider_name not in LLM_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
    
    # Create new LLM instance with the specified provider
    _current_llm = LLM_PROVIDERS[provider_name]()
    
    return _current_llm

# For backwards compatibility, keep the default LLM instance
llm = get_llm("gemini")
