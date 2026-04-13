import os
import requests
from dotenv import load_dotenv

load_dotenv()

FLOWISE_API_URL = os.getenv("FLOWISE_API_URL", "http://localhost:3000")


def query_flowise(question: str, chatflow_id: str = None, session_id: str = None) -> dict:
    """
    Calls the Flowise RAG API with a user question.

    Returns:
        {
            "response": str,              - LLM generated answer
            "retrieved_contexts": list    - Source documents retrieved by RAG
        }

    How it works (Real RAG Pipeline inside Flowise):
        User Query → Flowise Chatflow → Vector DB Retrieval (top-k docs)
        → LLM generates response using retrieved context
        → Returns response + sourceDocuments
    """
    chatflow_id = chatflow_id or os.getenv("FLOWISE_CHATFLOW_ID")
    if not chatflow_id:
        raise ValueError("FLOWISE_CHATFLOW_ID is not set in .env")

    url = f"{FLOWISE_API_URL}/api/v1/prediction/{chatflow_id}"

    payload = {"question": question}

    # Pass sessionId for multi-turn conversation support
    if session_id:
        payload["sessionId"] = session_id

    headers = {"Content-Type": "application/json"}

    # Add auth header only if Flowise API Key is configured
    flowise_api_key = os.getenv("FLOWISE_API_KEY")
    if flowise_api_key:
        headers["Authorization"] = f"Bearer {flowise_api_key}"

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()

    data = response.json()

    llm_response = data.get("text", "")

    # Extract retrieved source documents (the RAG retrieved contexts)
    source_documents = data.get("sourceDocuments", [])
    retrieved_contexts = [
        doc.get("pageContent", "")
        for doc in source_documents
        if doc.get("pageContent")
    ]

    # Fallback: if Flowise chatflow is not configured to return sourceDocuments
    if not retrieved_contexts:
        retrieved_contexts = [llm_response]

    return {
        "response": llm_response,
        "retrieved_contexts": retrieved_contexts
    }
