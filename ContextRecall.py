import pytest
from ragas import SingleTurnSample
from ragas.metrics import ContextRecall
from flowise_client import query_flowise

# -------------------------------------------------------------------
# CONTEXT RECALL
# Measures: Does the retrieved context cover all info in the reference answer?
# Score Range: 0 to 1 (higher is better)
# Threshold: >= 0.7 considered acceptable
# Requires: reference (ground truth answer)
# -------------------------------------------------------------------

# Replace it with queries and ground truth answers for YOUR Flowise knowledge base
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


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_context_recall(llm_wrapper, flowise_chatflow_id):
    context_recall = ContextRecall(llm=llm_wrapper)

    for test in TEST_DATA:
        # Real Flowise RAG API call
        result = query_flowise(question=test["user_input"], chatflow_id=flowise_chatflow_id)

        sample = SingleTurnSample(
            user_input=test["user_input"],
            retrieved_contexts=result["retrieved_contexts"],
            reference=test["reference"]
        )

        score = await context_recall.single_turn_ascore(sample)
        print(f"\n[Context Recall] Query: '{test['user_input']}' | Score: {score:.4f}")

        assert score >= 0.7, (
            f"Context Recall too low for query: '{test['user_input']}'\n"
            f"Score: {score:.4f} | Expected: >= 0.7\n"
            f"Retrieved Contexts: {result['retrieved_contexts']}"
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
    asyncio.run(test_context_recall(llm, chatflow_id))
