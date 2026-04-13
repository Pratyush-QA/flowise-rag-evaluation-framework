import pytest
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from flowise_client import query_flowise

# -------------------------------------------------------------------
# FAITHFULNESS
# Measures: Is the LLM response factually grounded in the retrieved context?
#           Detects hallucinations — claims NOT supported by retrieved docs.
# Score Range: 0 to 1 (higher is better)
# Threshold: >= 0.8 (stricter — hallucination is critical to catch)
# -------------------------------------------------------------------

# Replace it with queries relevant to YOUR Flowise knowledge base
TEST_QUERIES = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
]


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_faithfulness(llm_wrapper, flowise_chatflow_id):
    faithfulness = Faithfulness(llm=llm_wrapper)

    for query in TEST_QUERIES:
        # Real Flowise RAG API call
        result = query_flowise(question=query, chatflow_id=flowise_chatflow_id)

        sample = SingleTurnSample(
            user_input=query,
            response=result["response"],
            retrieved_contexts=result["retrieved_contexts"]
        )

        score = await faithfulness.single_turn_ascore(sample)
        print(f"\n[Faithfulness] Query: '{query}' | Score: {score:.4f}")

        assert score >= 0.8, (
            f"Faithfulness too low — possible hallucination detected for query: '{query}'\n"
            f"Score: {score:.4f} | Expected: >= 0.8\n"
            f"Response: {result['response']}\n"
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
    asyncio.run(test_faithfulness(llm, chatflow_id))
