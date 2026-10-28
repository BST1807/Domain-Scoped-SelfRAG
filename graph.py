"""graph.py
LangGraph construction and state routing for Domain-Scoped Self-RAG.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from nodes import (
    check_hallucination,
    check_usefulness,
    domain_guardrail,
    generate,
    refusal_node,
    retrieve,
    rewrite_query,
    self_rag_router,
)
from state import GraphState


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------
def route_domain(state: GraphState) -> Literal["self_rag_router", "refusal_node"]:
    """Conditional Edge: Directs to router if in-domain, or refusal if out-of-domain."""
    domain_result = state.get("domain_check_result", {})
    if domain_result.get("is_in_domain", False):
        return "self_rag_router"
    return "refusal_node"


def route_retrieval(state: GraphState) -> Literal["retrieve", "generate"]:
    """Conditional Edge: Directs to internal document retrieval or direct generation."""
    if state.get("retrieval_needed", False):
        return "retrieve"
    return "generate"


def route_post_generation(
    state: GraphState,
) -> Literal["check_hallucination", "check_usefulness"]:
    """Conditional Edge:
    If needs_retrieval == True -> check_hallucination
    If needs_retrieval == False -> check_usefulness
    """
    if state.get("retrieval_needed", False):
        return "check_hallucination"
    return "check_usefulness"


def route_hallucination(
    state: GraphState,
) -> Literal["generate", "check_usefulness"]:
    """Conditional Edge:
    If is_supported == False and loop_count < 2 -> generate (retry)
    If is_supported == True (or max loops reached) -> check_usefulness
    """
    is_supported = state.get("is_supported", False)
    hallucination_loops = state.get("hallucination_loop_count", 0)

    if not is_supported and hallucination_loops < 2:
        return "generate"
    return "check_usefulness"


def route_usefulness(state: GraphState) -> Literal["rewrite_query", "__end__"]:
    """Conditional Edge:
    If is_useful == False and loop_count < 2 -> rewrite_query -> retrieve
    If is_useful == True (or max loops reached) -> END
    """
    is_useful = state.get("is_useful", False)
    rewrite_loops = state.get("rewrite_loop_count", 0)

    if not is_useful and rewrite_loops < 2:
        return "rewrite_query"
    return END


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Builds and wires the Domain-Scoped Self-RAG StateGraph."""
    workflow = StateGraph(GraphState)

    # 1. Add all nodes
    workflow.add_node("domain_guardrail", domain_guardrail)
    workflow.add_node("refusal_node", refusal_node)
    workflow.add_node("self_rag_router", self_rag_router)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("check_usefulness", check_usefulness)
    workflow.add_node("rewrite_query", rewrite_query)

    # 2. Add entry point edge
    workflow.add_edge(START, "domain_guardrail")

    # 3. Add Domain Guardrail routing
    workflow.add_conditional_edges(
        "domain_guardrail",
        route_domain,
        {
            "self_rag_router": "self_rag_router",
            "refusal_node": "refusal_node",
        },
    )

    # 4. Out-of-domain refusal terminates at END
    workflow.add_edge("refusal_node", END)

    # 5. Add Self-RAG retrieval routing
    workflow.add_conditional_edges(
        "self_rag_router",
        route_retrieval,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # 6. Retrieved docs flow to generator
    workflow.add_edge("retrieve", "generate")

    # 7. Post-generation routing based on whether retrieval was used
    workflow.add_conditional_edges(
        "generate",
        route_post_generation,
        {
            "check_hallucination": "check_hallucination",
            "check_usefulness": "check_usefulness",
        },
    )

    # 8. Hallucination check loop (max 2 times) or advance to usefulness
    workflow.add_conditional_edges(
        "check_hallucination",
        route_hallucination,
        {
            "generate": "generate",
            "check_usefulness": "check_usefulness",
        },
    )

    # 9. Usefulness check loop (max 2 times) or terminate at END
    workflow.add_conditional_edges(
        "check_usefulness",
        route_usefulness,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        },
    )

    # 10. Rewritten query triggers new retrieval cycle
    workflow.add_edge("rewrite_query", "retrieve")

    return workflow


# Compile the graph for direct import and execution
app = build_graph().compile()
