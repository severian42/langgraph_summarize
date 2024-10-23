from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os



def get_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def get_ollama_embeddings():
    return OllamaEmbeddings(
        model="mxbai-embed-large:latest",  # You can change this to any model supported by Ollama
        base_url="http://localhost:11434",  # Adjust if your Ollama server is running elsewhere
    )

def get_huggingface_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # You can change this to any model you prefer
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

# Add more provider functions as needed, e.g.:
# def get_openai_embeddings():
#     return OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
