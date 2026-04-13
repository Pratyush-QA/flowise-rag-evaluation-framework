import pytest
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy
from flowise_client import query_flowise

# -------------------------------------------------------------------
# RESPONSE RELEVANCY
# Measures: Is the LLM response relevant and on-topic to the user query?
#           Low score = response went off-topic or was incomplete.
# Score Range: 0 to 1 (higher is better)
# Threshold: >= 0.7
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
async def test_response_relevancy(llm_wrapper, flowise_chatflow_id):
    response_relevancy = ResponseRelevancy(llm=llm_wrapper)

    for query in TEST_QUERIES:
        # Real Flowise RAG API call
        result = query_flowise(question=query, chatflow_id=flowise_chatflow_id)

        sample = SingleTurnSample(
            user_input=query,
            response=result["response"]
        )

        score = await response_relevancy.single_turn_ascore(sample)
        print(f"\n[Response Relevancy] Query: '{query}' | Score: {score:.4f}")

        assert score >= 0.7, (
            f"Response Relevancy too low for query: '{query}'\n"
            f"Score: {score:.4f} | Expected: >= 0.7\n"
            f"Response: {result['response']}"
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
    asyncio.run(test_response_relevancy(llm, chatflow_id))
