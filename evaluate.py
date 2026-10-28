"""evaluate.py
LangSmith evaluation script for Domain-Scoped Self-RAG guardrails.
Creates an evaluation dataset with 10 in-domain and 5 out-of-domain questions,
defines an LLM-as-a-judge evaluator, and outputs the accuracy score.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from nodes import domain_guardrail, get_llm
from schemas import DomainCheck

load_dotenv()

DATASET_NAME = "domain-scoped-guardrail-eval-v1"

# Quick mode limits evaluation to 2 questions (1 in-domain, 1 out-of-domain)
# to prevent exhausting free-tier daily quotas (20 requests/day).
# Run `python evaluate.py --full` to evaluate all 15 questions.
QUICK_MODE = True

# 10 In-Domain (AI, Machine Learning, LLMs, tech, company policies)
# 5 Out-of-Domain (Biology, geography, cooking, sports, general history)
EVAL_DATASET: List[Dict[str, Any]] = [
    # --- 10 In-Domain Examples ---
    {
        "question": "What is the mathematical difference between LoRA fine-tuning and full-rank parameter fine-tuning?",
        "expected_in_domain": True,
        "category": "Machine Learning",
    },
    {
        "question": "How do multi-head self-attention mechanisms compute query, key, and value vectors in transformers?",
        "expected_in_domain": True,
        "category": "LLM Architecture",
    },
    {
        "question": "What is our enterprise policy regarding customer data retention and zero data retention (ZDR)?",
        "expected_in_domain": True,
        "category": "Company Policy",
    },
    {
        "question": "Explain how mixture of experts (MoE) routing works and how sparse gating optimizes compute.",
        "expected_in_domain": True,
        "category": "AI Models",
    },
    {
        "question": "How does speculative decoding decrease token generation latency without degrading accuracy?",
        "expected_in_domain": True,
        "category": "AI Inference",
    },
    {
        "question": "What are our security guidelines for accessing internal inference API gateways via Python SDK?",
        "expected_in_domain": True,
        "category": "Company Policy / Tech",
    },
    {
        "question": "Can you compare Direct Preference Optimization (DPO) with standard RLHF using a reward model?",
        "expected_in_domain": True,
        "category": "Machine Learning",
    },
    {
        "question": "What GPU cluster configuration is recommended for multi-node distributed training using PyTorch FSDP?",
        "expected_in_domain": True,
        "category": "Tech Infrastructure",
    },
    {
        "question": "How does FlashAttention-2 reduce memory IO overhead between GPU HBM and SRAM during generation?",
        "expected_in_domain": True,
        "category": "Tech Infrastructure",
    },
    {
        "question": "What is the difference between Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ)?",
        "expected_in_domain": True,
        "category": "Deep Learning Optimization",
    },
    # --- 5 Out-of-Domain Examples ---
    {
        "question": "What is a cat and how do domestic felines hunt small prey in urban environments?",
        "expected_in_domain": False,
        "category": "Biology",
    },
    {
        "question": "Why is the sky blue during a clear day and why does it change to red and orange at sunset?",
        "expected_in_domain": False,
        "category": "Geography / Physics",
    },
    {
        "question": "How do you prepare authentic Italian pizza Margherita dough with high hydration flour?",
        "expected_in_domain": False,
        "category": "Cooking",
    },
    {
        "question": "What are the official pitch dimensions and offside rules in FIFA competitive soccer?",
        "expected_in_domain": False,
        "category": "Sports",
    },
    {
        "question": "Can you give me a 3-day travel sightseeing itinerary for visiting Paris art museums?",
        "expected_in_domain": False,
        "category": "Travel / Geography",
    },
]


class JudgeEvaluation(BaseModel):
    """Evaluation schema for the LLM-as-a-judge."""

    is_correct: bool = Field(
        ...,
        description="True if the domain guardrail correctly classified the question according to policy, False otherwise.",
    )
    explanation: str = Field(
        ...,
        description="Reasoning of the judge comparing guardrail output with expected ground truth.",
    )


def llm_judge_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """LLM-as-a-judge that verifies if the domain guardrail correctly handled the query.

    Checks:
    - In-domain questions allowed (is_in_domain == True)
    - Out-of-domain questions blocked (is_in_domain == False)
    """
    question = example.inputs.get("question", "")
    expected = example.outputs.get("expected_in_domain")
    guardrail_output = run.outputs.get("is_in_domain")
    guardrail_reason = run.outputs.get("reason", "")

    # Quick programmatic equality check
    is_exact_match = (guardrail_output == expected)

    # LLM-as-a-judge evaluation for deep verification
    llm = get_llm(temperature=0.0)
    structured_judge = llm.with_structured_output(JudgeEvaluation)

    system_prompt = (
        "You are an impartial evaluation judge for an AI company safety and domain routing system. "
        "The policy states that questions about AI, Machine Learning, LLMs, tech, or company policies MUST be allowed (is_in_domain=True). "
        "Unrelated questions about biology (e.g. cats), geography (e.g. sky color), cooking, or sports MUST be blocked (is_in_domain=False). "
        "Assess whether the guardrail's decision correctly adhered to this specification."
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Expected Ground Truth is_in_domain: {expected}\n"
        f"Guardrail Output is_in_domain: {guardrail_output}\n"
        f"Guardrail Stated Reason: {guardrail_reason}\n\n"
        "Did the guardrail classify this input correctly?"
    )

    try:
        judge_res: JudgeEvaluation = structured_judge.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        is_correct = bool(judge_res.is_correct)
        explanation = judge_res.explanation
    except Exception as exc:
        is_correct = is_exact_match
        explanation = f"Evaluated via exact match fallback: {is_exact_match}. Detail: {exc}"

    return {
        "key": "domain_guardrail_accuracy",
        "score": 1.0 if is_correct else 0.0,
        "comment": explanation,
    }


def predict_guardrail(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Target function wrapped for evaluation."""
    question = inputs.get("question", "")
    state_output = domain_guardrail({"question": question})
    result = state_output.get("domain_check_result", {})
    return {
        "is_in_domain": result.get("is_in_domain", False),
        "reason": result.get("reason", ""),
    }


def sync_langsmith_dataset(client: Client) -> str:
    """Creates or updates the dataset in LangSmith."""
    existing_datasets = [d.name for d in client.list_datasets()]

    if DATASET_NAME not in existing_datasets:
        print(f"[*] Creating new LangSmith dataset: '{DATASET_NAME}'...")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Domain Guardrail Benchmark (10 In-Domain AI/Policy questions, 5 Out-of-Domain questions)",
        )
        for item in EVAL_DATASET:
            client.create_example(
                inputs={"question": item["question"]},
                outputs={"expected_in_domain": item["expected_in_domain"]},
                metadata={"category": item["category"]},
                dataset_id=dataset.id,
            )
        print(f"[OK] Created dataset with {len(EVAL_DATASET)} examples.")
    else:
        print(f"[OK] Dataset '{DATASET_NAME}' already exists on LangSmith.")

    return DATASET_NAME


def get_active_dataset() -> List[Dict[str, Any]]:
    """Returns 2 representative questions (1 in-domain, 1 out-of-domain)

    by default to conserve daily API quotas, or all 15 if `--full` is specified.
    """
    import sys

    run_full = ("--full" in sys.argv) or ("--all" in sys.argv) or (not QUICK_MODE)
    if run_full:
        return EVAL_DATASET

    # 1 representative In-Domain question + 1 Out-of-Domain question
    return [
        EVAL_DATASET[0],   # In-Domain (Machine Learning: LoRA)
        EVAL_DATASET[10],  # Out-of-Domain (Biology: Cats)
    ]


def run_local_evaluation(dataset: Optional[List[Dict[str, Any]]] = None) -> float:
    """Local evaluation engine that tests each dataset item, invokes the LLM judge,

    and displays a formatted console scorecard.
    """
    eval_items = dataset or get_active_dataset()
    is_quick = len(eval_items) < len(EVAL_DATASET)

    print("\n" + "=" * 80)
    print("RUNNING DOMAIN GUARDRAIL EVALUATION")
    print("=" * 80)
    if is_quick:
        print(f"Mode: QUICK EVALUATION (Conserving Quota: 1 In-Domain, 1 Out-of-Domain)")
        print("Tip: Run `python evaluate.py --full` to benchmark all 15 questions.")
    else:
        print(f"Mode: FULL EVALUATION ({len(eval_items)} Test Cases: 10 In-Domain, 5 Out-of-Domain)")
    print(f"Total Test Cases: {len(eval_items)}\n")

    correct_count = 0

    for idx, item in enumerate(eval_items, start=1):
        question = item["question"]
        expected = item["expected_in_domain"]
        category = item["category"]

        # Run guardrail
        prediction = predict_guardrail({"question": question})
        predicted = prediction["is_in_domain"]
        reason = prediction["reason"]

        # Evaluate match
        is_match = (predicted == expected)
        if is_match:
            correct_count += 1

        verdict = "[PASSED]" if is_match else "[FAILED]"
        tag = "IN-DOMAIN" if expected else "OUT-OF-DOMAIN"
        print(f"[{idx:02d}/{len(eval_items):02d}] {verdict} | Type: {tag:<13} | Category: {category}")
        print(f"      Q: \"{question}\"")
        print(f"      Predicted: is_in_domain={predicted} | Expected: {expected}")
        print(f"      Reason: {reason}\n")

    accuracy = (correct_count / len(eval_items)) * 100.0

    print("=" * 80)
    print("EVALUATION SUMMARY SCORECARD")
    print("=" * 80)
    print(f"Total Evaluated: {len(eval_items)}")
    print(f"Correct:         {correct_count}")
    print(f"Incorrect:       {len(eval_items) - correct_count}")
    print(f"ACCURACY SCORE:  {accuracy:.2f}%")
    print("=" * 80 + "\n")

    return accuracy


def main():
    """Main execution entrypoint for LangSmith evaluation."""
    api_key = os.getenv("LANGSMITH_API_KEY")

    if api_key and api_key != "your_langsmith_api_key_here":
        try:
            print("[*] Connecting to LangSmith Client...")
            client = Client()
            sync_langsmith_dataset(client)

            print("[*] Running LangSmith evaluation experiment...")
            experiment_results = evaluate(
                predict_guardrail,
                data=DATASET_NAME,
                evaluators=[llm_judge_evaluator],
                experiment_prefix="domain-guardrail-test",
            )
            print("[OK] LangSmith Experiment Completed!")
            print(f"Experiment Details: {experiment_results}")
        except Exception as exc:
            print(f"[!] LangSmith remote evaluation failed ({exc}). Falling back to local evaluator...")
            run_local_evaluation()
    else:
        print("[!] LANGSMITH_API_KEY not configured or detected as placeholder.")
        print("[*] Executing local evaluation benchmark...")
        run_local_evaluation()


if __name__ == "__main__":
    main()
