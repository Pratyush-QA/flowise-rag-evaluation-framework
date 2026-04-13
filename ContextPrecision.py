import pytest
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from flowise_client import query_flowise

# -------------------------------------------------------------------
# CONTEXT PRECISION (without reference)
# Measures: Are the retrieved documents relevant to the user query?
# Score Range: 0 to 1 (higher is better)
# Threshold: >= 0.7 considered acceptable
# -------------------------------------------------------------------

# Replace these with queries relevant to YOUR Flowise knowledge base
TEST_QUERIES = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?"
]


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_context_precision(llm_wrapper, flowise_chatflow_id):
    context_precision = LLMContextPrecisionWithoutReference(llm=llm_wrapper)

    for query in TEST_QUERIES:
        # Real Flowise RAG API call — gets actual response + retrieved docs
        result = query_flowise(question=query, chatflow_id=flowise_chatflow_id)

        sample = SingleTurnSample(
            user_input=query,
            response=result["response"],
            retrieved_contexts=result["retrieved_contexts"]
        )

        score = await context_precision.single_turn_ascore(sample)
        print(f"\n[Context Precision] Query: '{query}' | Score: {score:.4f}")

        assert score >= 0.7, (
            f"Context Precision too low for query: '{query}'\n"
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
    asyncio.run(test_context_precision(llm, chatflow_id))
