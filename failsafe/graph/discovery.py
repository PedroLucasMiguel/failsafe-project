"""
Discovery graph — wires the Interview → Explorer → Analyzer → Indexer pipeline.

This is the main LangGraph StateGraph for the MVP discovery workflow.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from failsafe.agents.analyzer import analyzer_node
from failsafe.agents.explorer import explorer_node
from failsafe.agents.indexer import indexer_node
from failsafe.agents.interview import interview_node
from failsafe.state.discovery import DiscoveryState


def build_discovery_graph() -> StateGraph:
    """Construct and compile the discovery workflow graph.

    Flow: START → interview → explorer → analyzer → indexer → END

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    graph = StateGraph(DiscoveryState)

    # Add nodes
    graph.add_node("interview", interview_node)
    graph.add_node("explorer", explorer_node)
    graph.add_node("analyzer", analyzer_node)
    graph.add_node("indexer", indexer_node)

    # Wire edges
    graph.add_edge(START, "interview")
    graph.add_edge("interview", "explorer")
    graph.add_edge("explorer", "analyzer")
    graph.add_edge("analyzer", "indexer")
    graph.add_edge("indexer", END)

    return graph.compile()
