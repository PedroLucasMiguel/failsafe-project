"""
Coding graph — LangGraph workflow for the coding agent.

Flow: START → coder → END

The coding agent handles its own tool loop internally (ReAct via bind_tools),
so the graph is minimal: just a single node that runs CodingAgent.run().
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from failsafe.state.coding import CodingState


def coder_node(state: CodingState) -> dict:
    """Run the coding agent for the given task."""
    from failsafe.agents.coder import CodingAgent
    from failsafe.knowledge.store import KnowledgeStore
    from failsafe.knowledge.vector_store import VectorStore
    from failsafe.llm import get_llm

    llm = get_llm()
    codebase_path = Path(state["codebase_path"])

    # Load KB
    kb_json = codebase_path / "failsafe_kb.json"
    import json
    kb_store = KnowledgeStore()
    if kb_json.exists():
        kb_store.load_from_dict(json.loads(
            kb_json.read_text(encoding="utf-8")))

    # Attach vector store if available
    vs = None
    vs_path = codebase_path / ".failsafe" / "vectors"
    if vs_path.exists():
        vs = VectorStore(vs_path)
        kb_store.attach_vector_store(vs)

    agent = CodingAgent(
        llm=llm,
        kb_store=kb_store,
        vector_store=vs,
        codebase_path=codebase_path,
    )

    result = agent.run(state["task"])

    return {
        "files_modified": result.files_modified,
        "impact":         result.impact,
        "summary":        result.summary,
        "kb_context":     state.get("kb_context", ""),
    }


def build_coding_graph() -> StateGraph:
    """Build and compile the coding agent workflow."""
    graph = StateGraph(CodingState)
    graph.add_node("coder", coder_node)
    graph.add_edge(START, "coder")
    graph.add_edge("coder", END)
    return graph.compile()
