import os
import json
import nltk
import pytest
from datetime import datetime
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    FactualCorrectness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall
)
from flowise_client import query_flowise

# -------------------------------------------------------------------
# MULTI-METRIC EVALUATION — LOCAL REPORTS
# Runs all 5 key RAG metrics in one shot against Flowise API.
# Results are saved locally to reports/ folder as CSV and JSON.
#
# Metrics evaluated:
#   1. FactualCorrectness   - Is the answer factually correct?
#   2. ResponseRelevancy    - Is the answer relevant to the question?
#   3. Faithfulness         - Is the answer grounded in retrieved context?
#   4. ContextPrecision     - Are retrieved docs relevant to the query?
#   5. ContextRecall        - Do retrieved docs cover the reference answer?
# -------------------------------------------------------------------

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

# Replace with queries + ground truth for YOUR Flowise knowledge base
TEST_DATA = [
    {
        "user_input": "Who introduced the theory of relativity?",
        "reference": "Albert Einstein introduced the theory of relativity, which transformed our understanding of time, space, and gravity."
    },
    {
        "user_input": "Who was the first computer programmer?",
        "reference": "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's Analytical Engine."
    },
    {
        "user_input": "What did Isaac Newton contribute to science?",
        "reference": "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics."
    }
]

# Quality gate thresholds
THRESHOLDS = {
    "factual_correctness": 0.7,
    "response_relevancy": 0.7,
    "faithfulness": 0.8,
    "llm_context_precision_without_reference": 0.7,
    "context_recall": 0.7
}


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_multi_metrics(llm_wrapper, flowise_chatflow_id):
    metrics = [
        FactualCorrectness(llm=llm_wrapper),
        ResponseRelevancy(llm=llm_wrapper),
        Faithfulness(llm=llm_wrapper),
        LLMContextPrecisionWithoutReference(llm=llm_wrapper),
        LLMContextRecall(llm=llm_wrapper)
    ]

    samples = []
    for test in TEST_DATA:
        # Real Flowise RAG API call for each test query
        result = query_flowise(question=test["user_input"], chatflow_id=flowise_chatflow_id)
        print(f"\n[Flowise] Query: '{test['user_input']}'\n"
              f"          Response: {result['response'][:100]}...\n"
              f"          Retrieved {len(result['retrieved_contexts'])} context(s)")

        samples.append(SingleTurnSample(
            user_input=test["user_input"],
            response=result["response"],
            retrieved_contexts=result["retrieved_contexts"],
            reference=test["reference"]
        ))

    eval_dataset = EvaluationDataset(samples)
    result = evaluate(metrics=metrics, dataset=eval_dataset)

    print(f"\n{'='*60}")
    print(f"RAGAS Multi-Metric Evaluation Results")
    print(f"{'='*60}")
    print(result)

    # ── Save results locally ──────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save full per-sample results as CSV
    scores = result.to_pandas()
    csv_path = os.path.join(REPORTS_DIR, f"multi_metrics_{timestamp}.csv")
    scores.to_csv(csv_path, index=False)
    print(f"\n[Report] CSV saved → {csv_path}")

    # 2. Save summary (average scores) as JSON
    summary = {}
    for metric_name in THRESHOLDS:
        if metric_name in scores.columns:
            summary[metric_name] = round(float(scores[metric_name].mean()), 4)
    summary["timestamp"] = timestamp
    summary["total_samples"] = len(scores)

    json_path = os.path.join(REPORTS_DIR, f"multi_metrics_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Report] JSON saved → {json_path}")

    # 3. Print summary table
    print(f"\n{'─'*45}")
    print(f"{'Metric':<45} {'Avg Score':>10} {'Threshold':>10} {'Status':>8}")
    print(f"{'─'*45}")
    for metric_name, threshold in THRESHOLDS.items():
        if metric_name in scores.columns:
            avg_score = scores[metric_name].mean()
            status = "✅ PASS" if avg_score >= threshold else "❌ FAIL"
            print(f"{metric_name:<45} {avg_score:>10.4f} {threshold:>10.1f} {status:>8}")
    print(f"{'─'*45}")

    # ── Quality gate assertions ───────────────────────────────────
    for metric_name, threshold in THRESHOLDS.items():
        if metric_name in scores.columns:
            avg_score = scores[metric_name].mean()
            assert avg_score >= threshold, (
                f"{metric_name} below threshold!\n"
                f"Average Score: {avg_score:.4f} | Expected: >= {threshold}"
            )


if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    load_dotenv()
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    chatflow_id = os.getenv("FLOWISE_CHATFLOW_ID")
    asyncio.run(test_multi_metrics(llm, chatflow_id))
