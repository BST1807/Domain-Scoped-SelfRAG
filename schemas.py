"""schemas.py
Pydantic schemas for the Domain-Scoped Self-RAG Knowledge Assistant.
"""

from pydantic import BaseModel, Field


class DomainCheck(BaseModel):
    """Binary decision whether the user question is within the allowed AI company domain."""

    is_in_domain: bool = Field(
        ...,
        description=(
            "True if the user asks about AI, Machine Learning, LLMs, tech, or company policies. "
            "False if the user asks about unrelated topics like biology (e.g., cats), "
            "geography (e.g., sky color), or cooking."
        ),
    )
    reason: str = Field(
        ...,
        description="A concise rationale explaining why the question is classified as in-domain or out-of-domain.",
    )


class ShouldRetrieve(BaseModel):
    """Decision on whether internal proprietary document retrieval is required."""

    needs_retrieval: bool = Field(
        ...,
        description=(
            "True if internal document retrieval is needed to answer specific proprietary policies, "
            "internal architecture specs, or enterprise guidelines. False if the question can be answered "
            "using general AI/tech parametric knowledge."
        ),
    )
    reason: str = Field(
        ...,
        description="Concise rationale explaining why retrieval is or is not required.",
    )


class IsRelevant(BaseModel):
    """Document relevance assessment against the user query."""

    is_relevant: bool = Field(
        ...,
        description="True if the retrieved document chunk is directly relevant and helpful for the user query, False otherwise.",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why the document is or is not relevant.",
    )


class IsSupported(BaseModel):
    """Hallucination check evaluating if the generation is grounded in the retrieved documents."""

    is_supported: bool = Field(
        ...,
        description=(
            "True if the generated answer is completely grounded in and supported by the retrieved "
            "documents without making unsupported or hallucinated factual claims. False otherwise."
        ),
    )
    reason: str = Field(
        ...,
        description="Detailed explanation of whether the claims are supported or if hallucination occurred.",
    )


class IsUseful(BaseModel):
    """Usefulness check evaluating whether the answer effectively resolves the user's question."""

    is_useful: bool = Field(
        ...,
        description="True if the answer directly, accurately, and completely resolves the user's question, False otherwise.",
    )
    reason: str = Field(
        ...,
        description="Explanation of why the answer is useful or what key aspects were missed.",
    )
