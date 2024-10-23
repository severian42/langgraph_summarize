from typing import List, TypedDict, Any
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents a state of a graph.

    Attributes:
        question: Question
        retriever: Document retriever
        generation: LLM Generation
        use_web_search: whether to use web search
        documents: List of documents
        documents_relevant: whether documents are relevant
    """

    question: str
    retriever: Any
    generation: str
    use_web_search: bool
    documents: List[Document]
    documents_relevant: bool
