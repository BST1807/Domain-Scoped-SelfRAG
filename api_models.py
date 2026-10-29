"""api_models.py
Pydantic schemas for the FastAPI microservice.
Defines request and response data contracts for the Self-RAG service.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Input payload for the /chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="The query or prompt submitted by the user.",
        examples=["What is our machine learning model deployment policy?"],
    )


class ChatResponse(BaseModel):
    """Output payload returned by the /chat endpoint."""

    answer: str = Field(
        ...,
        description="The generated response from the Domain-Scoped Self-RAG pipeline.",
    )
    latency: float = Field(
        ...,
        description="Total end-to-end request latency in seconds.",
        examples=[1.245],
    )
    model_version: str = Field(
        ...,
        description="Active LLM model version and provider used to service the request.",
        examples=["gemini-2.5-flash"],
    )
