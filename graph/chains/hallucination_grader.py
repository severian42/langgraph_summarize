import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from models.llm import get_llm

# Load environment variables
load_dotenv()

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generated answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )

def get_hallucination_grader():
    llm = get_llm()
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    message = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", message),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    return hallucination_prompt | structured_llm_grader

hallucination_grader = get_hallucination_grader()
