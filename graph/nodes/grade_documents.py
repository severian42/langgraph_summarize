from typing import Any, Dict
from graph.state import GraphState
from graph.chains import retrieval_grader

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the user question.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): Dictionary containing filtered relevant documents and relevance status
    """
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    relevant_documents = []
    for doc in documents:
        result = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if result.binary_score.lower() == "yes":
            print("---DOCUMENT IS RELEVANT---")
            relevant_documents.append(doc)
        else:
            print("---DOCUMENT IS NOT RELEVANT---")

    documents_relevant = len(relevant_documents) > 0
    return {
        "documents": relevant_documents,
        "documents_relevant": documents_relevant,
        "question": question
    }
