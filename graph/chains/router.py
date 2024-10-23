from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from models.llm import get_llm


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Route the user query to either 'vectorstore' or 'web_search'",
    )


def get_router_chain():
    llm = get_llm()
    structured_llm_router = llm.with_structured_output(RouteQuery)

    message = """You are an expert at routing user questions to the appropriate data source.
        The vectorstore contains documents related to machine learning concepts such as: agents, prompt engineering, and adversarial attacks.
        Route to:
        - 'vectorstore' for questions about ML concepts, agents, prompting, etc.
        - 'web_search' for current events, specific facts, or topics not in the vectorstore"""

    router_prompt = ChatPromptTemplate.from_messages(
        [("system", message), ("human", "{question}")]
    )

    return router_prompt | structured_llm_router


question_router = get_router_chain()
