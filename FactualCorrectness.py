import pytest
from ragas import SingleTurnSample
from ragas.metrics import FactualCorrectness
from flowise_client import query_flowise

# -------------------------------------------------------------------
# FACTUAL CORRECTNESS
# Measures: Is the LLM response factually correct compared to reference?
#           Checks for accuracy of claims, not just style.
# Score Range: 0 to 1 (higher is better)
# Threshold: >= 0.7
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
        "user_input": "Who won two Nobel Prizes for research on radioactivity?",
        "reference": "Marie Curie won two Nobel Prizes for her pioneering research on radioactivity."
    }
]


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_factual_correctness(llm_wrapper, flowise_chatflow_id):
    factual_correctness = FactualCorrectness(llm=llm_wrapper)

    for test in TEST_DATA:
        # Real Flowise RAG API call
        result = query_flowise(question=test["user_input"], chatflow_id=flowise_chatflow_id)

        sample = SingleTurnSample(
            user_input=test["user_input"],
            response=result["response"],
            reference=test["reference"]
        )

        score = await factual_correctness.single_turn_ascore(sample)
        print(f"\n[Factual Correctness] Query: '{test['user_input']}' | Score: {score:.4f}")

        assert score >= 0.65, (
            f"Factual Correctness too low for query: '{test['user_input']}'\n"
            f"Score: {score:.4f} | Expected: >= 0.65\n"
            f"Response: {result['response']}\n"
            f"Reference: {test['reference']}"
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
    asyncio.run(test_factual_correctness(llm, chatflow_id))
