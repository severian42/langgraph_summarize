import sys
import os
from dotenv import load_dotenv
import gradio as gr
from graph.graph import create_graph
from models.llm import get_llm, LLM_PROVIDERS
from models.em import get_embedding_model, EMBEDDING_PROVIDERS
from ingestion import ingest_documents, get_retriever

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = os.getenv('USER_AGENT', 'agentic_rag/1.0')

# Print loaded environment variables (for debugging)
print("Loaded environment variables:")
for key in ['TAVILY_API_KEY', 'LANGCHAIN_API_KEY', 'LANGCHAIN_TRACING_V2', 'LANGCHAIN_PROJECT', 'GOOGLE_API_KEY', 'HUGGINGFACE_API_KEY', 'OPENAI_API_KEY', 'USER_AGENT']:
    value = os.getenv(key)
    print(f"{key}: {'*' * len(value) if value else 'Not set'}")

def process_question(question, llm_provider, embedding_provider):
    # Set the LLM and embedding model based on user selection
    llm = get_llm(llm_provider)
    embedding_model = get_embedding_model(embedding_provider)
    
    # Get the latest retriever (including any newly ingested documents)
    retriever = get_retriever()
    
    # Create the workflow graph
    workflow_app = create_graph()
    
    try:
        # Initialize the state with required fields
        initial_state = {
            "question": question,
            "retriever": retriever,
            "documents": [],
            "generation": "",
            "documents_relevant": False
        }
        
        # Process the question using the workflow
        result = workflow_app.invoke(initial_state)
        
        # Extract the final answer from the result
        final_answer = result.get("generation", "No answer generated.")
        
        # Prepare a detailed output
        output = f"Question: {question}\n\n"
        output += f"LLM Provider: {llm_provider}\n"
        output += f"Embedding Provider: {embedding_provider}\n\n"
        output += f"Answer: {final_answer}\n\n"
        
        # Add debug information if available
        if "documents" in result:
            output += "Retrieved Documents:\n"
            for i, doc in enumerate(result["documents"], 1):
                output += f"Document {i}: {doc.page_content[:200]}...\n\n"
        
        if "use_web_search" in result:
            output += f"Web Search Used: {result['use_web_search']}\n"
        
    except Exception as e:
        output = f"An error occurred while processing the question: {str(e)}"
        print(f"Error details: {e}")  # Add detailed error logging
    
    return output

def upload_and_ingest(files):
    if not files:
        return "No files uploaded."
    
    try:
        ingest_documents(files)
        return f"Successfully ingested {len(files)} document(s)."
    except Exception as e:
        return f"Error ingesting documents: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=process_question,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
        gr.Dropdown(choices=list(LLM_PROVIDERS.keys()), value="gemini", label="LLM Provider"),
        gr.Dropdown(choices=list(EMBEDDING_PROVIDERS.keys()), value="gemini", label="Embedding Provider")
    ],
    outputs=gr.Textbox(lines=10, label="Answer and Debug Info"),
    title="Agentic RAG System",
    description="Ask a question and get an answer using the Agentic RAG system.",
    examples=[
        ["What is agent memory in context of LLMs?", "gemini", "gemini"],
        ["Explain the concept of prompt engineering.", "ollama", "ollama"],
        ["What are adversarial attacks on language models?", "huggingface", "huggingface"]
    ]
)

# Add document upload functionality
upload_interface = gr.Interface(
    fn=upload_and_ingest,
    inputs=gr.File(file_count="multiple", label="Upload Documents"),
    outputs=gr.Textbox(label="Ingestion Status"),
    title="Document Upload",
    description="Upload documents to be ingested into the RAG system."
)

# Combine the interfaces
combined_interface = gr.TabbedInterface([iface, upload_interface], ["Ask Questions", "Upload Documents"])

if __name__ == "__main__":
    combined_interface.launch()
