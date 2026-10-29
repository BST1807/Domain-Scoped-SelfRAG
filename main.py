"""main.py
FastAPI microservice serving the LangGraph Domain-Scoped Self-RAG agent.
Provides RESTful endpoints for health verification and question-answering.
"""

import logging
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api_models import ChatRequest, ChatResponse
from graph import app as rag_agent

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enterprise-rag-api")

# Initialize FastAPI application
app = FastAPI(
    title="Domain-Scoped Self-RAG Microservice",
    description=(
        "Production-grade FastAPI service exposing the LangGraph Self-RAG "
        "enterprise knowledge assistant with domain guardrails, adaptive retrieval, "
        "and hallucination/usefulness verification."
    ),
    version="1.0.0",
)

# Enable CORS for cross-origin client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_version() -> str:
    """Resolves active LLM model version based on provider configuration."""
    load_dotenv(override=True)
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider in ("gemini", "google"):
        return os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
    return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify microservice readiness and liveness."""
    return {
        "status": "healthy",
        "service": "Domain-Scoped Self-RAG Microservice",
        "model_version": get_model_version(),
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Executes the Domain-Scoped Self-RAG LangGraph pipeline for an incoming user question.

    Measures execution latency and captures errors safely within a try/except block.
    """
    start_time = time.perf_counter()
    model_version = get_model_version()

    initial_state: Dict[str, Any] = {
        "question": request.question,
        "documents": [],
        "generation": "",
        "domain_check_result": {},
        "retrieval_needed": False,
        "loop_count": 0,
        "hallucination_loop_count": 0,
        "rewrite_loop_count": 0,
        "is_supported": False,
        "is_useful": False,
        "thought_process": [],
    }

    try:
        final_state = rag_agent.invoke(initial_state)
        raw_answer = final_state.get(
            "generation",
            "I am an AI company assistant. I could not generate a response for your query.",
        )
        answer = _format_answer(raw_answer)
    except Exception as exc:
        logger.error("Error executing Self-RAG graph: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing Self-RAG graph: {str(exc)}",
        ) from exc

    latency = round(time.perf_counter() - start_time, 4)

    return ChatResponse(
        answer=answer,
        latency=latency,
        model_version=model_version,
    )


def _format_answer(raw_answer: Any) -> str:
    """Extracts clean text if the model returned structured chunks or stringified dicts."""
    import ast

    if isinstance(raw_answer, str):
        trimmed = raw_answer.strip()
        if trimmed.startswith(("{'type'", '{"type"')):
            try:
                parsed = ast.literal_eval(trimmed)
                if isinstance(parsed, dict) and "text" in parsed:
                    return str(parsed["text"]).strip()
            except Exception:
                pass
        return trimmed

    if isinstance(raw_answer, dict):
        return str(raw_answer.get("text", raw_answer)).strip()

    if isinstance(raw_answer, list):
        parts = []
        for item in raw_answer:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts).strip()

    return str(raw_answer).strip()
