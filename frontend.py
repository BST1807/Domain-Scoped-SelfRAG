"""frontend.py
Streamlit chat interface for the Domain-Scoped Self-RAG Knowledge Assistant.
Visualizes internal routing decisions and thought process via st.expander.
"""

import importlib
import os
import sys
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Ensure .env settings take precedence
load_dotenv(override=True)

# Force hot-reload of graph and nodes modules in long-running Streamlit processes
import nodes
importlib.reload(nodes)
import graph
importlib.reload(graph)
from graph import app

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Domain-Scoped Self-RAG | Enterprise Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .thought-item {
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 6px;
        background-color: rgba(30, 41, 59, 0.4);
        border-left: 3px solid #3b82f6;
        font-family: monospace;
        font-size: 0.88rem;
    }
    .status-badge-passed {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-badge-failed {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=64)
    st.markdown("### 🏢 Enterprise AI Guardrails")
    st.markdown(
        "This assistant implements a **Domain-Scoped Self-RAG** architecture "
        "powered by **LangGraph**, **LangChain**, and **Pydantic**."
    )

    st.markdown("---")
    st.markdown("#### ⚙️ LLM Provider & Config")

    provider = st.selectbox(
        "Select Provider",
        options=["Google Gemini", "OpenAI"],
        index=0,
    )

    if provider == "Google Gemini":
        os.environ["LLM_PROVIDER"] = "gemini"
        default_gemini_key = os.getenv("GEMINI_API_KEY", "")
        if "active_gemini_key" not in st.session_state or st.session_state.get("_last_loaded_env_key") != default_gemini_key:
            st.session_state["active_gemini_key"] = default_gemini_key
            st.session_state["_last_loaded_env_key"] = default_gemini_key

        gemini_key_input = st.text_input(
            "Gemini API Key",
            value=st.session_state["active_gemini_key"],
            type="password",
            placeholder="AIza... or AQ...",
            help="Configured in .env as GEMINI_API_KEY",
        )
        if gemini_key_input:
            os.environ["GEMINI_API_KEY"] = gemini_key_input
            st.session_state["active_gemini_key"] = gemini_key_input
            os.environ.pop("GOOGLE_API_KEY", None)

        selected_model = st.selectbox(
            "Gemini Model",
            options=["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.5-flash", "gemini-pro-latest"],
            index=0,
        )
        os.environ["GEMINI_MODEL_NAME"] = selected_model
    else:
        os.environ["LLM_PROVIDER"] = "openai"
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=default_openai_key,
            type="password",
            placeholder="sk-...",
            help="Configured in .env as OPENAI_API_KEY",
        )
        if openai_key_input:
            os.environ["OPENAI_API_KEY"] = openai_key_input

        selected_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )
        os.environ["OPENAI_MODEL_NAME"] = selected_model

    # LangSmith tracing status & project navigation
    langsmith_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGSMITH_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "domain-scoped-self-rag")
        st.success("🟢 LangSmith Tracing Active", icon="✅")
        st.markdown(
            "📊 **[Open Traces in LangSmith ↗](https://smith.langchain.com/o/cc07bb80-93d1-40f8-8fc5-e62ec1055fa7/projects/p/fbf04951-bb89-44ed-8575-8e46764bd55f)**\n\n"
            "<span style='font-size:0.8rem;color:#94a3b8;'>Target Project: <code>domain-scoped-self-rag</code></span>",
            unsafe_allow_html=True,
        )
    else:
        st.info("⚪ LangSmith Tracing Idle (Set LANGSMITH_API_KEY in .env)")

    st.markdown("---")
    st.markdown("#### 🧪 Test Prompts")
    st.caption("Click a sample question to test guardrails & routing:")

    col1, col2 = st.columns(2)
    sample_prompt = None

    if st.button("AI Fine-tuning (In-Domain)", use_container_width=True):
        sample_prompt = "What is the difference between LoRA and full parameter fine-tuning?"

    if st.button("Company Policy (In-Domain)", use_container_width=True):
        sample_prompt = "What is our company's policy on customer data retention and zero data retention (ZDR)?"

    if st.button("Sky Color (Out-of-Domain)", use_container_width=True):
        sample_prompt = "Why is the sky blue?"

    if st.button("Cats (Out-of-Domain)", use_container_width=True):
        sample_prompt = "What is a cat and how do they hunt?"

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("active_gemini_key", None)
        st.session_state.pop("_last_loaded_env_key", None)
        load_dotenv(override=True)
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Domain-Scoped Self-RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Enterprise knowledge retrieval strictly scoped to AI, ML, tech, and company policies '
    'with automated hallucination and usefulness verification loops.'
    '</div>',
    unsafe_allow_html=True,
)

active_p = "Google Gemini" if os.getenv("LLM_PROVIDER", "gemini") == "gemini" else "OpenAI"
active_m = (
    os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
    if active_p == "Google Gemini"
    else os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
)

st.markdown(
    f'<div style="margin-top:-15px; margin-bottom: 20px; font-size: 0.9rem; color: #38bdf8;">'
    f'⚡ Connected Engine: <strong>{active_p}</strong> (<code>{active_m}</code>)'
    f'</div>',
    unsafe_allow_html=True,
)

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your AI Company Knowledge Assistant. I can assist with "
                "AI/ML architecture, LLM systems, engineering guidelines, and internal company policies. "
                "How may I help you today?"
            ),
            "thought_process": [
                "System initialized: Domain guardrail active.",
                "Policy scope: In-domain questions allowed; out-of-domain topics blocked.",
            ],
        }
    ]

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        # Crucial Requirement: st.expander("Agent Thought Process")
        if msg.get("thought_process"):
            with st.expander("Agent Thought Process", expanded=(msg["role"] == "assistant")):
                for thought in msg["thought_process"]:
                    st.markdown(f"- {thought}")
        st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Chat Input & Workflow Execution
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask a question about AI, ML, tech, or company policies...")
if sample_prompt:
    user_input = sample_prompt

if user_input:
    # 1. Append and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Invoke the LangGraph workflow
    with st.chat_message("assistant"):
        with st.spinner("Analyzing domain scope and routing through Self-RAG..."):
            initial_state: Dict[str, Any] = {
                "question": user_input,
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
                final_state = app.invoke(initial_state)
                generation = final_state.get(
                    "generation",
                    "I am an AI company assistant. I can only answer questions related to our domain.",
                )
                thought_process = final_state.get("thought_process", [])
            except Exception as exc:
                generation = f"An unexpected error occurred during workflow execution: {exc}"
                thought_process = [f"⚠️ Execution Exception: {exc}"]

        # 3. Crucial Requirement: Render Agent Thought Process expander
        with st.expander("Agent Thought Process", expanded=True):
            if thought_process:
                for thought in thought_process:
                    st.markdown(f"- {thought}")
            else:
                st.write("No thought process captured.")

        # 4. Display final response
        st.markdown(generation)

    # 5. Persist assistant response in session state
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": generation,
            "thought_process": thought_process,
        }
    )
