from abc import ABC, abstractmethod
from app.graph_state import GraphState

class BaseAgentNode(ABC):
    """Abstract base class for all agent nodes in the multi-agent graph."""

    @abstractmethod
    def execute(self, state: GraphState) -> GraphState:
        """Execute the agent's logic on the given state."""
        pass

    def __call__(self, state: GraphState) -> GraphState:
        """Allows the instance to be called directly, maintaining LangGraph node compatibility."""
        return self.execute(state)
