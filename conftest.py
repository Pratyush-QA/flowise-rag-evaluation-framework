import os
import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

load_dotenv()


@pytest.fixture(scope="session")
def llm_wrapper():
    """Shared LLM fixture used across all RAGAS evaluation tests."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return LangchainLLMWrapper(llm)


@pytest.fixture(scope="session")
def flowise_chatflow_id():
    """Reads chatflow ID from .env. Skips test if not configured."""
    chatflow_id = os.getenv("FLOWISE_CHATFLOW_ID")
    if not chatflow_id:
        pytest.skip("FLOWISE_CHATFLOW_ID not set in .env — skipping Flowise test")
    return chatflow_id
