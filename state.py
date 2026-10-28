"""state.py
State definition for the Domain-Scoped Self-RAG Knowledge Assistant.
"""

import operator
from typing import Annotated, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """The central state of the Domain-Scoped Self-RAG workflow.

    Attributes:
        question: The user's input question or current rewritten query.
        documents: List of retrieved context documents (strings).
        generation: The generated response text from the LLM.
        domain_check_result: Dictionary holding the domain verification outcome
            ({'is_in_domain': bool, 'reason': str}).
        retrieval_needed: Boolean indicating whether internal document retrieval is needed.
        loop_count: General iteration counter for workflow cycles.
        hallucination_loop_count: Number of hallucination retry iterations (max 2).
        rewrite_loop_count: Number of query rewrite retry iterations (max 2).
        is_supported: Whether the generated answer is grounded in retrieved documents.
        is_useful: Whether the generated answer satisfies the user's question.
        thought_process: Chronological execution trace and routing log
            displayed in the Streamlit UI thought process expander.
    """

    question: str
    documents: list[str]
    generation: str
    domain_check_result: dict
    retrieval_needed: bool
    loop_count: int
    hallucination_loop_count: int
    rewrite_loop_count: int
    is_supported: bool
    is_useful: bool
    thought_process: Annotated[list[str], operator.add]
