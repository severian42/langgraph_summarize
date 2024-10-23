import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from models.llm import get_llm

# Load environment variables
load_dotenv()

class GradeAnswer(BaseModel):
    """Binary score for assessing the answer is relevant to the question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'."
    )

def get_answer_grader():
    llm = get_llm()
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    message = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", message),
            ("human", "User question: {question} \n\n LLM generated answer: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader

answer_grader = get_answer_grader()
