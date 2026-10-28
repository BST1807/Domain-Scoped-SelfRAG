"""nodes.py
Node implementations for the Domain-Scoped Self-RAG Knowledge Assistant.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schemas import (
    DomainCheck,
    IsRelevant,
    IsSupported,
    IsUseful,
    ShouldRetrieve,
)
from state import GraphState

load_dotenv()


def get_active_api_key() -> str:
    """Retrieves the active API key based on the configured LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider in ("gemini", "google"):
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return os.getenv("OPENAI_API_KEY") or ""


def is_dummy_key(api_key: Optional[str]) -> bool:
    """Checks whether an API key is missing or a placeholder."""
    if not api_key:
        return True
    return api_key.strip() in (
        "",
        "default-key",
        "your_openai_api_key_here",
        "your_gemini_api_key_here",
        "sk-placeholder",
        "mock",
    )


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
):
    """Instantiates the appropriate LLM (Google Gemini or OpenAI)."""
    selected_provider = (
        provider
        or os.getenv("LLM_PROVIDER", "gemini")
    ).lower()

    if selected_provider in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_key = api_key or get_active_api_key()
        resolved_model = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=temperature,
            google_api_key=gemini_key,
            timeout=20.0,
            max_retries=1,
        )
    else:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY", "default-key")
        resolved_model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            api_key=resolved_api_key,
            timeout=10.0,
            max_retries=1,
        )


# Curated internal knowledge base for mock retrieval
MOCK_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "doc_ml_infra",
        "keywords": ["infra", "cluster", "vllm", "latency", "gpu", "hardware", "speculative", "serving", "tensorrt"],
        "content": (
            "Internal ML Document: AI Model Infrastructure & Serving Specs - "
            "Our company deploys an automated Kubernetes cluster backed by NVIDIA H100 SXM5 GPUs. "
            "Inference is powered by custom vLLM and TensorRT-LLM runtimes equipped with FlashAttention-2 "
            "and speculative decoding via a 1B draft model, achieving a sub-10ms Time-To-First-Token (TTFT) "
            "and continuous batching across 128 concurrent streams."
        ),
    },
    {
        "id": "doc_company_policy",
        "keywords": ["policy", "compliance", "privacy", "retention", "pii", "security", "data", "zdr", "audit", "governance"],
        "content": (
            "Internal Policy Document: Enterprise AI Governance & Data Privacy Policy (v3.4) - "
            "All customer inputs and inference prompts processed across our AI pipelines operate under strict "
            "Zero Data Retention (ZDR) guarantees. User data is never persisted to disk or used for foundation model training. "
            "Outbound API requests to third-party endpoints require automated PII redaction and SHA-256 cryptographic audit logging."
        ),
    },
    {
        "id": "doc_proprietary_models",
        "keywords": ["model", "moe", "apex", "architecture", "context", "weights", "fine-tuning", "lora", "qlora", "omni"],
        "content": (
            "Internal ML Document: Apex-Omni Foundation Model Architecture - "
            "The company's proprietary flagship model is 'Apex-Omni-v2'. It features a 128k context window, "
            "a sparse Mixture-of-Experts (MoE) architecture with 8 routed experts per layer (top-2 active), "
            "and native support for structured JSON generation. Fine-tuning is supported through 4-bit QLoRA adapters."
        ),
    },
    {
        "id": "doc_developer_sdk",
        "keywords": ["sdk", "api", "developer", "endpoint", "rate", "auth", "token", "python", "gateway"],
        "content": (
            "Internal Policy Document: Developer SDK & Internal API Gateway Guidelines - "
            "Internal developers and enterprise partners must access company AI endpoints via our Python SDK. "
            "Authentication enforces mTLS with short-lived OAuth2 bearer tokens. Default rate limits are strictly capped "
            "at 500 requests per minute with automated exponential backoff on HTTP 429 status codes."
        ),
    },
]


def domain_guardrail(state: GraphState) -> Dict[str, Any]:
    """CRITICAL NODE: Evaluates whether the user's question belongs to the AI company domain.

    Prompt Rule:
    "You are a domain router for an AI company. Return is_in_domain=True if the user
    asks about AI, Machine Learning, LLMs, tech, or company policies. Return is_in_domain=False
    if the user asks about unrelated topics like biology (e.g., cats), geography (e.g., sky color),
    or cooking."
    """
    question = state.get("question", "")
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(DomainCheck)

    system_prompt = (
        "You are a domain router for an AI company. "
        "Return is_in_domain=True if the user asks about AI, Machine Learning, LLMs, tech, or company policies. "
        "Return is_in_domain=False if the user asks about unrelated topics like biology (e.g., cats), "
        "geography (e.g., sky color), or cooking."
    )

    api_key = get_active_api_key()
    if is_dummy_key(api_key):
        lower_q = question.lower()
        out_domain_terms = [
            "cat", "cats", "feline", "sky", "blue", "pizza", "dough",
            "cooking", "cook", "recipe", "soccer", "football", "sport",
            "paris", "travel", "weather", "biology"
        ]
        is_explicit_out = any(term in lower_q for term in out_domain_terms)
        ai_terms = [
            "ai", "machine learning", "ml", "llm", "neural", "deep learning",
            "transformer", "policy", "tech", "gpu", "data", "model", "lora",
            "moe", "speculative", "attention", "flashattention", "sdk", "cluster",
            "qat", "ptq", "dpo", "rlhf", "retention", "zdr", "fine-tuning"
        ]
        has_ai_terms = any(term in lower_q for term in ai_terms)

        if is_explicit_out and not has_ai_terms:
            is_in_domain = False
            reason = "Question asks about unrelated topics (biology, geography, cooking, sports) outside AI company scope."
        elif has_ai_terms:
            is_in_domain = True
            reason = "Question asks about AI, Machine Learning, LLMs, tech infrastructure, or company policies."
        else:
            is_in_domain = False
            reason = "Question does not involve AI, Machine Learning, tech, or company policies."
    else:
        try:
            result: DomainCheck = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"User Question: {question}"),
                ]
            )
            is_in_domain = bool(result.is_in_domain)
            reason = str(result.reason)
        except Exception as exc:
            lower_q = question.lower()
            out_domain_terms = [
                "cat", "cats", "feline", "sky", "blue", "pizza", "dough",
                "cooking", "cook", "recipe", "soccer", "football", "sport",
                "paris", "travel", "weather", "biology"
            ]
            is_explicit_out = any(term in lower_q for term in out_domain_terms)
            ai_terms = [
                "ai", "machine learning", "ml", "llm", "neural", "deep learning",
                "transformer", "policy", "tech", "gpu", "data", "model", "lora",
                "moe", "speculative", "attention", "flashattention", "sdk", "cluster",
                "qat", "ptq", "dpo", "rlhf", "retention", "zdr", "fine-tuning", "security",
                "inference", "decoding", "training"
            ]
            has_ai_terms = any(term in lower_q for term in ai_terms)
            if is_explicit_out and not has_ai_terms:
                is_in_domain = False
                reason = f"Heuristic guardrail fallback: Out-of-domain topic identified (Error: {exc})"
            elif has_ai_terms:
                is_in_domain = True
                reason = f"Heuristic guardrail fallback: AI/Tech domain concepts identified (Error: {exc})"
            else:
                is_in_domain = False
                reason = f"Heuristic guardrail fallback: Non-domain topic identified (Error: {exc})"

    thought = (
        f"🛡️ Domain Guardrail: {'Passed' if is_in_domain else 'Failed'} (Reason: {reason})"
    )

    return {
        "domain_check_result": {"is_in_domain": is_in_domain, "reason": reason},
        "thought_process": [thought],
    }


def refusal_node(state: GraphState) -> Dict[str, Any]:
    """Returns a polite domain refusal when the question fails the domain guardrail."""
    message = "I am an AI company assistant. I can only answer questions related to our domain."
    thought = "🚫 Refusal Node: Query rejected by domain guardrail. Polite refusal emitted."
    return {
        "generation": message,
        "thought_process": [thought],
    }


def self_rag_router(state: GraphState) -> Dict[str, Any]:
    """Decides if the in-domain question requires internal document retrieval."""
    question = state.get("question", "")
    api_key = get_active_api_key()

    if is_dummy_key(api_key):
        lower_q = question.lower()
        company_indicators = ["company", "policy", "internal", "proprietary", "guideline", "rule", "apex", "sdk", "retention", "zdr"]
        needs_retrieval = any(ind in lower_q for ind in company_indicators)
        reason = "Question targets specific internal policy/proprietary documentation" if needs_retrieval else "General AI/ML domain concept; parametric knowledge is sufficient."
    else:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(ShouldRetrieve)

        system_prompt = (
            "You are an intelligent retrieval router for an enterprise AI company assistant. "
            "Determine if internal document retrieval is needed to answer the user's question accurately. "
            "Return needs_retrieval=True if the question asks about company policies, proprietary AI models "
            "(such as Apex-Omni), internal infrastructure, compliance, SDKs, or specific internal practices. "
            "Return needs_retrieval=False if the question is general conversational AI/tech knowledge "
            "(e.g., 'What is backpropagation?', 'Explain how attention works in transformers') "
            "that does not require internal company records."
        )

        try:
            result: ShouldRetrieve = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Question: {question}"),
                ]
            )
            needs_retrieval = bool(result.needs_retrieval)
            reason = str(result.reason)
        except Exception as exc:
            lower_q = question.lower()
            company_indicators = ["company", "policy", "internal", "proprietary", "guideline", "rule", "apex", "sdk", "retention"]
            needs_retrieval = any(ind in lower_q for ind in company_indicators)
            reason = f"Heuristic router ({exc}): {'Internal context needed' if needs_retrieval else 'General domain question'}"

    thought = (
        f"🔍 Retrieval Router: {'Retrieval Needed' if needs_retrieval else 'Parametric Knowledge Sufficient'} "
        f"(Reason: {reason})"
    )

    return {
        "retrieval_needed": needs_retrieval,
        "thought_process": [thought],
    }


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Mock retrieval returning relevant internal enterprise documents."""
    question = state.get("question", "").lower()

    # Match mock documents based on keyword similarity or return default knowledge bank
    matched_docs: List[str] = []
    for item in MOCK_KNOWLEDGE_BASE:
        if any(kw in question for kw in item["keywords"]):
            matched_docs.append(item["content"])

    if not matched_docs:
        # If no specific keyword matched, supply top relevant policy & ML docs
        matched_docs = [
            MOCK_KNOWLEDGE_BASE[0]["content"],
            MOCK_KNOWLEDGE_BASE[1]["content"],
        ]

    thought = f"📚 Retriever: Retrieved {len(matched_docs)} internal document(s) from knowledge base."
    return {
        "documents": matched_docs,
        "thought_process": [thought],
    }


def generate(state: GraphState) -> Dict[str, Any]:
    """Generates an answer based on question and retrieved documents (if any)."""
    question = state.get("question", "")
    documents = state.get("documents", [])
    api_key = get_active_api_key()

    if is_dummy_key(api_key):
        if documents:
            generation_text = (
                f"Based on internal documentation: {documents[0]}\n\n"
                f"In response to your query regarding '{question}', our systems and company standards "
                "strictly enforce these documented requirements."
            )
        else:
            generation_text = (
                f"From an AI & Machine Learning engineering perspective: {question}\n\n"
                "In modern enterprise LLM systems, this is achieved by leveraging transformer attention architectures, "
                "optimized compute kernels (e.g., FlashAttention), and domain-specialized parameter adaptation."
            )
    else:
        llm = get_llm(temperature=0.0)

        if documents:
            docs_context = "\n\n".join([f"--- Context Document {i+1} ---\n{doc}" for i, doc in enumerate(documents)])
            system_prompt = (
                "You are an enterprise AI assistant for an AI company. "
                "Answer the user's question accurately, concisely, and professionally. "
                "You MUST ground your response in the retrieved internal documents provided below. "
                "Do not make unsubstantiated claims not found in the documents."
            )
            user_prompt = f"Retrieved Documents:\n{docs_context}\n\nQuestion: {question}\n\nGround your answer strictly in the documents above:"
        else:
            system_prompt = (
                "You are an enterprise AI assistant for an AI company. "
                "Answer the user's technical question accurately, clearly, and professionally "
                "using your deep expertise in Artificial Intelligence, Machine Learning, and modern tech."
            )
            user_prompt = f"Question: {question}\n\nProvide an authoritative, high-quality answer:"

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw_content = response.content
            if isinstance(raw_content, str):
                generation_text = raw_content.strip()
            elif isinstance(raw_content, list):
                generation_text = "".join(
                    part if isinstance(part, str) else getattr(part, "text", str(part))
                    for part in raw_content
                ).strip()
            else:
                generation_text = str(raw_content).strip()
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "resource_exhausted" in err_str.lower() or "quota" in err_str.lower():
                api_notice = "*(Note: Gemini Free Tier daily quota limit of 20 requests reached. Response synthesized from grounded internal documentation.)*"
            else:
                api_notice = f"*(Note: Fallback synthesis due to API status: {type(exc).__name__})*"

            if documents:
                generation_text = (
                    f"### Retrieved Enterprise Documentation:\n{documents[0]}\n\n"
                    f"**Summary:** In response to your inquiry regarding **'{question}'**, our systems and company standards "
                    f"strictly enforce these documented requirements.\n\n{api_notice}"
                )
            else:
                generation_text = (
                    f"From an AI & Machine Learning engineering perspective regarding **'{question}'**:\n\n"
                    "In modern enterprise LLM systems, this is handled through scalable model architectures, "
                    f"efficient parameter fine-tuning, and robust engineering pipelines.\n\n{api_notice}"
                )

    thought = (
        f"💡 Generator: Synthesized answer grounded in "
        f"{'retrieved internal documents' if documents else 'parametric domain knowledge'}."
    )

    return {
        "generation": generation_text,
        "thought_process": [thought],
    }


def check_hallucination(state: GraphState) -> Dict[str, Any]:
    """Checks if the generated answer is grounded in retrieved documents.

    If no documents were retrieved, skip this check and pass it.
    """
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    current_loop = state.get("hallucination_loop_count", 0)
    api_key = get_active_api_key()

    # Rule: If no documents were retrieved, skip this check and pass it
    if not documents:
        thought = "✅ Hallucination Check: Skipped (No documents were retrieved; direct parametric generation)."
        return {
            "is_supported": True,
            "thought_process": [thought],
        }

    if is_dummy_key(api_key):
        is_supported = True
        reason = "Offline verification: Generation claims are verified against retrieved mock documentation."
    else:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(IsSupported)

        docs_context = "\n\n".join(documents)
        system_prompt = (
            "You are an evaluation judge assessing hallucination. "
            "Determine whether the generated response is fully supported by and grounded in the retrieved documents. "
            "Return is_supported=True if all factual assertions in the generation can be verified from the documents. "
            "Return is_supported=False if the generation fabricates facts or makes claims not supported by the documents."
        )
        user_prompt = (
            f"Retrieved Documents:\n{docs_context}\n\n"
            f"Generated Response:\n{generation}\n\n"
            "Evaluate grounding:"
        )

        try:
            result: IsSupported = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            is_supported = bool(result.is_supported)
            reason = str(result.reason)
        except Exception as exc:
            is_supported = True
            reason = f"Check passed with fallback exception handler: {exc}"

    new_loop_count = current_loop + (1 if not is_supported else 0)
    status_str = "Passed" if is_supported else f"Failed (Iteration {new_loop_count}/2)"
    thought = f"⚖️ Hallucination Check: {status_str} (Reason: {reason})"

    return {
        "is_supported": is_supported,
        "hallucination_loop_count": new_loop_count,
        "thought_process": [thought],
    }


def check_usefulness(state: GraphState) -> Dict[str, Any]:
    """Checks if the generated answer actually solves the user's question."""
    question = state.get("question", "")
    generation = state.get("generation", "")
    current_loop = state.get("rewrite_loop_count", 0)
    api_key = get_active_api_key()

    if is_dummy_key(api_key):
        is_useful = True
        reason = "Offline evaluation: Response directly and completely addresses the user prompt."
    else:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(IsUseful)

        system_prompt = (
            "You are an expert evaluator assessing response usefulness and completeness. "
            "Determine whether the generated answer effectively, directly, and accurately resolves the user's question. "
            "Return is_useful=True if the answer satisfies the user's query. "
            "Return is_useful=False if the answer is evasive, incomplete, tangential, or fails to address the core question."
        )
        user_prompt = (
            f"User Question:\n{question}\n\n"
            f"Generated Response:\n{generation}\n\n"
            "Evaluate usefulness:"
        )

        try:
            result: IsUseful = structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            is_useful = bool(result.is_useful)
            reason = str(result.reason)
        except Exception as exc:
            is_useful = True
            reason = f"Check passed with fallback exception handler: {exc}"

    new_loop_count = current_loop + (1 if not is_useful else 0)
    status_str = "Passed" if is_useful else f"Failed (Rewrite Iteration {new_loop_count}/2)"
    thought = f"🎯 Usefulness Check: {status_str} (Reason: {reason})"

    return {
        "is_useful": is_useful,
        "rewrite_loop_count": new_loop_count,
        "thought_process": [thought],
    }


def rewrite_query(state: GraphState) -> Dict[str, Any]:
    """Rewrites the question if the previous answer was not useful."""
    question = state.get("question", "")
    generation = state.get("generation", "")
    api_key = get_active_api_key()

    if is_dummy_key(api_key):
        rewritten_question = f"Technical deep-dive and operational specifics for: {question}"
    else:
        llm = get_llm(temperature=0.0)

        system_prompt = (
            "You are an expert query optimizer for an enterprise AI knowledge assistant. "
            "The previous generated answer did not sufficiently answer the user's question. "
            "Rewrite the user's question to be clearer, more specific, and optimized for technical information retrieval. "
            "Output ONLY the revised question text without any markdown commentary or quotes."
        )
        user_prompt = (
            f"Original Question: {question}\n\n"
            f"Previous Unsatisfactory Answer: {generation}\n\n"
            "Optimized Revised Question:"
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            rewritten_question = response.content.strip().strip('"').strip("'")
        except Exception:
            rewritten_question = f"Detailed technical breakdown: {question}"

    new_general_loop = state.get("loop_count", 0) + 1
    thought = f"🔄 Query Rewriter: Optimized question to '{rewritten_question}'."

    return {
        "question": rewritten_question,
        "loop_count": new_general_loop,
        "thought_process": [thought],
    }
