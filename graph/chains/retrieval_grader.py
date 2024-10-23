from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from models.llm import get_llm


class GradeDocuments(BaseModel):
    """Binary score for the relevance check of retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question? 'yes' or 'no'",
    )


def get_retrieval_grader():
    llm = get_llm()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    message = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", message),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader


retrieval_grader = get_retrieval_grader()
