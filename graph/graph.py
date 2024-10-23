from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.state import GraphState
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS
from graph.chains import hallucination_grader, answer_grader
from graph.nodes import generate, grade_documents, retrieve

load_dotenv()

def create_graph():
    """Create the workflow graph."""
    
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(GENERATE, generate)

    # Define the edges
    # Start with retrieval
    workflow.set_entry_point(RETRIEVE)
    
    # After retrieval, grade the documents
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    
    # After grading, either generate or retrieve more based on relevance
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        # Function that determines the next node
        lambda x: GENERATE if x["documents_relevant"] else RETRIEVE,
        # Map condition values to nodes
        {
            GENERATE: GENERATE,
            RETRIEVE: RETRIEVE
        }
    )
    
    # After generation, check for hallucinations and relevance
    def check_generation(state):
        print("---CHECK HALLUCINATIONS AND RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # Check if generation is grounded in documents
        hallucination_score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        
        if hallucination_score.binary_score.lower() == "yes":
            # If grounded, check if it answers the question
            answer_score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            if answer_score.binary_score.lower() == "yes":
                return "end"
            else:
                return RETRIEVE
        return RETRIEVE

    # Add conditional edges from GENERATE
    workflow.add_conditional_edges(
        GENERATE,
        check_generation,
        {
            "end": END,
            RETRIEVE: RETRIEVE
        }
    )
    
    # Compile the graph
    return workflow.compile()

# Optionally, you can add this to generate a visual representation of the graph
app = create_graph()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
