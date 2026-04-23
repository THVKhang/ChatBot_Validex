from langgraph.graph import StateGraph, END
from app.graph_state import GraphState
from app.agents.parser_node import parser_node
from app.agents.researcher_node import researcher_node
from app.agents.writer_node import writer_node
from app.agents.editor_node import editor_node

def route_after_editor(state: GraphState):
    """Conditional edge from Editor: If feedback exists, go back to Writer. Else END."""
    if state.get("editor_feedback"):
        return "Writer"
    return END

# Build Graph
builder = StateGraph(GraphState)

# Add Nodes
builder.add_node("Parser", parser_node)
builder.add_node("Researcher", researcher_node)
builder.add_node("Writer", writer_node)
builder.add_node("Editor", editor_node)

# Set Edges
builder.set_entry_point("Parser")
builder.add_edge("Parser", "Researcher")
builder.add_edge("Researcher", "Writer")
builder.add_edge("Writer", "Editor")

# Conditional Edge
builder.add_conditional_edges("Editor", route_after_editor)

# Compile Graph
multi_agent_graph = builder.compile()
