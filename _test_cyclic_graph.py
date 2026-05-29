import sys; sys.path.insert(0, '.')
from app.graph import multi_agent_graph
from app.session_manager import SessionManager
import logging

logging.basicConfig(level=logging.INFO)

def test_graph():
    session = SessionManager()
    initial_state = {
        "prompt": "explain what is validex",
        "session": session,
        "request_id": "test_loop",
        "revision_count": 0,
        "retrieval_attempts": 0,
        "tried_queries": []
    }
    
    print("Testing Graph Execution...")
    try:
        # Run graph
        final_state = multi_agent_graph.invoke(initial_state)
        print("\n=== GRAPH COMPLETED ===")
        print(f"Retrieval Attempts: {final_state.get('retrieval_attempts')}")
        print(f"RAG Feedback: {final_state.get('rag_feedback')}")
        print(f"Title: {final_state.get('title')}")
    except Exception as e:
        print(f"Error executing graph: {e}")

if __name__ == "__main__":
    test_graph()
