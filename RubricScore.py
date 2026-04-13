import pytest
from ragas import SingleTurnSample
from ragas.metrics import RubricsScore
from flowise_client import query_flowise

# -------------------------------------------------------------------
# RUBRIC SCORE
# Measures: Custom criteria-based scoring of the LLM response.
#           You define what score 1-5 means for YOUR use case.
# Score Range: 1 to 5 (higher is better)
# Threshold: >= 4 (meaning response is mostly accurate and well-aligned)
# -------------------------------------------------------------------

# Define rubric criteria relevant to your use case
RUBRICS = {
    "score1_description": "The response is incorrect, irrelevant, or does not align with the expected answer.",
    "score2_description": "The response partially matches but includes significant errors or irrelevant information.",
    "score3_description": "The response generally aligns but lacks detail or has minor inaccuracies.",
    "score4_description": "The response is mostly accurate with only minor issues or missing details.",
    "score5_description": "The response is fully accurate, clear, detailed, and completely aligned with the expected answer."
}

# Replace it with queries relevant to YOUR Flowise knowledge base
# Also update the reference ground truth answers accordingly
TEST_DATA = [
    {
        "user_input": "Where is the Eiffel Tower located?",
        "reference": "The Eiffel Tower is located in Paris, France."
    },
    {
        "user_input": "Who introduced the theory of relativity?",
        "reference": "Albert Einstein introduced the theory of relativity."
    }
]


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_rubric_score(llm_wrapper, flowise_chatflow_id):
    rubrics_score = RubricsScore(rubrics=RUBRICS, llm=llm_wrapper)

    for test in TEST_DATA:
        # Real Flowise RAG API call
        result = query_flowise(question=test["user_input"], chatflow_id=flowise_chatflow_id)

        sample = SingleTurnSample(
            user_input=test["user_input"],
            response=result["response"],
            reference=test["reference"]
        )

        score = await rubrics_score.single_turn_ascore(sample=sample)
        print(f"\n[Rubric Score] Query: '{test['user_input']}' | Score: {score}/5")

        assert score >= 4, (
            f"Rubric Score too low for query: '{test['user_input']}'\n"
            f"Score: {score}/5 | Expected: >= 4\n"
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
    asyncio.run(test_rubric_score(llm, chatflow_id))
