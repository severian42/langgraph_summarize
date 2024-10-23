from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.llms import HuggingFaceHub
import os



def get_gemini_llm(google_api_key=None):
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", google_api_key=google_api_key
    )

def get_ollama_llm():
    return ChatOllama(
        model="vanilj/supernova-medius:q8_0",  # You can change this to any model supported by Ollama
        base_url="http://localhost:11434",  # Adjust if your Ollama server is running elsewhere
    )

def get_huggingface_llm():
    return HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",  # You can change this to any model you prefer
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

# Add more provider functions as needed, e.g.:
# def get_openai_llm():
#     return ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"])
