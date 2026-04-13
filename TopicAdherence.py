import uuid
import pytest
from ragas import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import TopicAdherenceScore
from flowise_client import query_flowise

# -------------------------------------------------------------------
# TOPIC ADHERENCE (Multi-Turn)
# Measures: Does the AI stay on topic across a multi-turn conversation?
#           Uses Flowise sessionId to maintain conversation context.
# Score Range: 0 to 1 (higher is better)
# Threshold: >= 0.7
# Mode: "precision" — focuses on how often AI responded on-topic
# -------------------------------------------------------------------

# Multi-turn conversation scenario
# The AI should stay on science topics and flag when user goes off-topic
CONVERSATION_TURNS = [
    "Can you provide me with details about Einstein's theory of relativity?",
    "Tell me more about general relativity.",
    "What other contributions did Einstein make to physics?",
    # Off-topic turn below — topic adherence should detect this
    "By the way, do you know any good recipes for a chocolate cake?"
]

# Topics the conversation is expected to stay within
REFERENCE_TOPICS = ["science", "physics", "Einstein"]


@pytest.mark.asyncio
@pytest.mark.flowise
@pytest.mark.ragas
async def test_topic_adherence(llm_wrapper, flowise_chatflow_id):
    topic_adherence = TopicAdherenceScore(llm=llm_wrapper, mode="precision")

    # Use a unique sessionId so Flowise maintains conversation context across turns
    session_id = str(uuid.uuid4())
    conversation_messages = []

    for turn_query in CONVERSATION_TURNS:
        # Real Flowise RAG API call with sessionId for multi-turn context
        result = query_flowise(
            question=turn_query,
            chatflow_id=flowise_chatflow_id,
            session_id=session_id
        )

        conversation_messages.append(HumanMessage(content=turn_query))
        conversation_messages.append(AIMessage(content=result["response"]))
        print(f"\n[Turn] Q: {turn_query}\n       A: {result['response'][:100]}...")

    sample = MultiTurnSample(
        user_input=conversation_messages,
        reference_topics=REFERENCE_TOPICS
    )

    score = await topic_adherence.multi_turn_ascore(sample)
    print(f"\n[Topic Adherence] Score: {score:.4f} | Topics: {REFERENCE_TOPICS}")

    assert score >= 0.7, (
        f"Topic Adherence too low\n"
        f"Score: {score:.4f} | Expected: >= 0.7\n"
        f"Reference Topics: {REFERENCE_TOPICS}"
    )


if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    chatflow_id = os.getenv("FLOWISE_CHATFLOW_ID")
    asyncio.run(test_topic_adherence(llm, chatflow_id))
