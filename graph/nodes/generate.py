from typing import Any, Dict

from graph.state import GraphState
from graph.chains.generation import get_generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response to the user question.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the generated response and the question
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Prepare context with enhanced citations
    context_with_citations = []
    for i, doc in enumerate(documents, 1):
        # Extract source metadata if available
        source = getattr(doc.metadata, 'source', f'doc_{i}')
        page = getattr(doc.metadata, 'page', '1')
        
        # Format citation with source information
        citation = f"[source={source}&page={page}]"
        context_with_citations.append(f"{doc.page_content} {citation}")
    
    context = "\n\n".join(context_with_citations)
    
    generation_chain = get_generation_chain()
    generation = generation_chain.invoke({
        "CHUNKS": context,
        "QUESTION": question
    })
    
    return {
        "generation": generation,
        "documents": documents,
        "question": question
    }
