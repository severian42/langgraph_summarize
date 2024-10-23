from typing import Any, Dict
from graph.state import GraphState

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]
    
    documents = retriever.get_relevant_documents(question)
    
    return {
        "documents": documents,
        "question": question
    }
